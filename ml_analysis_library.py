import jsonpickle
import os

from option_parsers import get_logger
from settings import LoggingSettings, DataFormat, Paths, Order, Result

logger = get_logger(LoggingSettings.LOGGING_LEVEL.value, LoggingSettings.OUTPUT_FILE.value,
                    "ml_analysis_library")
from typing import Dict, List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from parsing_library import JobMetadata, get_parsed_data_by_test, count_deviation_for_row, \
    get_datasets_for_types


class MLTestCaseOutput:
    """
    This class represents output class of TestCase from Beaker.
    """

    def __init__(self, test_id, type_label: str, correct_result=False, valid=None, threshold=None,
                 result_value=None):
        self.type_label = type_label
        self.test_id = test_id
        self.valid = valid
        self.correct_result = correct_result
        self.threshold = threshold
        if result_value is not None:
            self.result_value = float(result_value)

    def as_nice_string(self) -> str:
        if not self.correct_result:
            return "ML analysis for test id {} type of {} wasn't done, because something went " \
                   "wrong.\n".format(self.test_id, self.type_label)
        if self.valid:
            return "Seems that entered dataset for test id {} type of {} is valid and " \
                   "no anomaly founded. Threshold is {} and deviation was {}\n".format(self.test_id,
                                                                                       self.type_label,
                                                                                       self.threshold,
                                                                                       self.result_value)
        return "Anomaly found for dataset for test id {} type of {}.Threshold is {} and deviation was {}\n".format(
            self.test_id,
            self.type_label,
            self.threshold, self.result_value)


class MLResultOutput:
    """
    This class represents output of the ML analysis.
    """

    def __init__(self, job_id, test_results=None):
        if test_results is None:
            test_results = []
        self.job_id = job_id
        self.test_results = test_results

    def add_result(self, test_id, label, correct_result=False, valid=None, threshold=None,
                   result_value=None):
        self.test_results.append(
            MLTestCaseOutput(test_id, label, correct_result, valid, threshold,
                             result_value))

    def as_json(self):
        return jsonpickle.encode(self, unpicklable=False)

    def as_nice_string(self) -> str:
        return_string = "ML output for job with ID:\n"
        for test_result in self.test_results:
            return_string += test_result.as_nice_string()
        return return_string


def execute_ml_for_single_input(model_path: str, threshold_path: str, input_data: List[List[
    float]]) -> \
        (bool, float, float):
    """
    This function is responsible for executing ML analysis for single output. It executes analysis
    by model and returns result, threshold and deviation.
    :param model_path: This param is path to ML model.
    :param threshold_path: This param is path to file with single float threshold.
    :param input_data: Data for ML analysis for single row, size of nested list is by number of
    dimensions -> [[1.2,0.4],[0.25,0.6]]
    :return: Tuple, first element says if dataset is Ok or is it anomaly. Second parameter is
    deviation and third threshold.
    """
    logger.debug("Starting to make analysis for model: {} and threshold: {}".format(model_path,
                                                                                    threshold_path))
    model = keras.models.load_model(model_path)  # we loads ML model
    threshold = float(open(threshold_path).readline())  # we loads threshold value from the file
    predicted_data = model.predict([input_data])
    # model returns result for input data, it can accepts entries from more datasets, so we need to
    # use list parenthesis
    deviation = count_deviation_for_row(input_data, predicted_data[0])
    # we gets deviation from output and input data
    return_bool = True
    if deviation > threshold:
        return_bool = False
    logger.debug("Successfully ending single analysis.")
    return return_bool, deviation, threshold


def sort_dimensions_of_data(test_dataset: Dict[str, float], order: List[str]) -> List[List[float]]:
    """
    This function gets dataset as dictionary and labels order as list. Then creates multidimensional
    list in the provided order.
    Example: test_dataset : {"b":[1,2,4,5],"a":[1,4,5,6]}
             order: ["a","b"]
             -> [[1,1],[4,2],[5,4],[6,5]]
    """
    logger.debug("Starting to sort data by order: {}".format(order))
    multidimensional_array = []
    for index in range(DataFormat.ROW_NUMBER.value):
        time_sample = []
        for key in order:
            time_sample.append(test_dataset[key][index])
        multidimensional_array.append(time_sample)
    logger.debug("End of sorting data.")
    return multidimensional_array


def check_by_model(datasets: Dict, data_dir: str, nodes: int, result_metadata: MLResultOutput) \
        -> bool:
    """
    This function gets datasets parsed from pcp_output, data directory with models and data for ML
    , number of nodes and output structure. Then prepare data in correct format for all types and
    testcases, execute analysis and saves output information for each of them.
    :param datasets: Dictionary with all datasets by types and testcases.
            -> {'mem_available': { 'testcase_1': { 'mem_ava_alert':[0.5,0.6] }},
                'cpus': { 'testcase_1': { 'ksmd_alert':[0.4,0.2], 'cpu_idle':[0.2,0.1] }}
                }
    :param data_dir: Root dir with data for ML and with models. -> './data'
    :param nodes: Number of nodes -> 2
    :param result_metadata: Result class, where we will put data for output.
    :return:
    """
    logger.debug("Starting to get results from ML models for  data_dir: {}, nodes: {}"
                 .format(data_dir, nodes))
    for ml_type in datasets.keys():
        for test in datasets[ml_type].keys():
            # we get paths, that we need
            model_path = Paths.MODEL_PATH.value.format(data_dir, ml_type, nodes, test)
            threshold_path = Paths.THRESHOLD_PATH.value.format(data_dir, ml_type, nodes, test)
            try:
                # we get List with data from the dictionary in format set in settings
                input_data = sort_dimensions_of_data(datasets[ml_type][test],
                                                     Order.ORDER.value[ml_type])
                # we execute ML analysis with that data
                is_valid, deviation, threshold = execute_ml_for_single_input(model_path,
                                                                             threshold_path,
                                                                             input_data)
            except Exception as e:
                logger.error("Loading ML and threshold for test id {} type of {} failed because of "
                             "error: {}"
                             .format(test, ml_type, e))
                result_metadata.add_result(test, ml_type)
                # we record that we tried to make analysis, but problem occurred with ML
                continue
            # we log results to metadata_object
            result_metadata.add_result(test, ml_type, True, is_valid, threshold, deviation)
    logger.debug("Successfully ending to get results from ML models.")
    return True


def get_data_for_single_output(file_path: str, test_list: List[str], skip_cpus: bool,
                               skip_memory: bool,
                               result: Result =
                    Result.CORRECT.value):
    """
    This function gets path to pcp_output and other options and parse PCP data to format for
    ML analysis.
    :param file_path: Path to pcp_output which we want to parse and get data from.
    :param test_list: List with names of testcases, that we want to get. Can be empty.
    :param skip_cpus: Boolean value, which indicates if we want to skip data for type cpus.
    :param skip_memory: Boolean value, which indicates if we want to skip data for type memory.
    :param result: Parameter if result is valid or not.
    :return: Return dictionary in format
             -> {'mem_available': { 'testcase_1': { 'mem_ava_alert':[0.5,0.6] }},
                'cpus': { 'testcase_1': { 'ksmd_alert':[0.4,0.2], 'cpu_idle':[0.2,0.1] }}
                }
    """
    logger.debug("Starting to get data for ML for path: {}.".format(file_path))
    return_output = {}
    job_metadata = JobMetadata()
    list_with_datasets = get_datasets_for_types(file_path, test_list, skip_cpus,
                                                skip_memory, job_metadata,
                                                result)
    for dataset in list_with_datasets:
        return_output[dataset.get_name()] = get_parsed_data_by_test(dataset, job_metadata)
    logger.debug("Successfully ending getting data for ML.")
    return return_output
