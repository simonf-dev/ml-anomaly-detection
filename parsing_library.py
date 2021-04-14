import json
import math
import os
import pandas as pd
from option_parsers import get_logger
from settings import LoggingSettings

logger = get_logger(LoggingSettings.LOGGING_LEVEL.value, LoggingSettings.OUTPUT_FILE.value,
                    "parsing_library")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from typing import List, Dict, TextIO, KeysView
from settings import ConstantValues, InputDataLabels, PlotNames, Result, DataFormat, Order, Paths


class StealError(Exception):
    def __init__(self, message=None):
        if message:
            self.message = message
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return self.message
        else:
            return "StealError has been raised."


class NoDataError(Exception):
    def __init__(self, message=None):
        if message:
            self.message = message
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return self.message
        else:
            return "NoDataError error has been raised."


class FinalDataOutput:
    def __init__(self, name: str, data: Dict[str, List[float]],
                 result: Result = Result.CORRECT.value):

        self._name: str = name

        self._data: Dict[str: List[float]] = data

        self._result: Result = result

    def get_time(self, time: int) -> float or None:
        try:
            return self._data[InputDataLabels.TIME.value].index(time)
        except:
            return None

    def get_value_by_label_and_index(self, label: str, index: int):
        return self._data[label][index]

    def get_values_by_label(self, label: str):
        return self._data[label]

    def get_values(self):
        return self._data

    def set_value_on_label_and_index(self, label: str, index: int, value: float):
        self._data[label][index] = value

    def delete_dataset(self, label: str):
        del self._data[label]

    def get_name(self):
        return self._name

    def get_result(self):
        return self._result


class Plot:
    def __init__(self, label: str, data: str):
        self._label: str = label
        self._data: Dict[str, List[float or None]] = parse_single_charts_from_chart_set(data)
        for metric in self._data:
            normalize_none_values_from_array(self._data[metric])

    def get_time(self, time: int) -> float or None:
        try:
            return self._data[InputDataLabels.TIME.value].index(time)
        except ValueError:
            return None

    def get_value_by_label_and_index(self, label: str, index: int) -> float or None:
        try:
            return self._data[label][int(index)]
        except ValueError:
            return None

    def get_values(self) -> Dict[str, List[float or None]]:
        return self._data

    def get_values_by_label(self, label: str) -> List[float or None]:
        return self._data[label]

    def set_value_on_label_and_index(self, label: str, index: int, value: float):
        self._data[label][index] = value

    def get_data_labels(self) -> KeysView[str]:
        return self._data.keys()

    def delete_dataset(self, label: str):
        del self._data[label]

    def get_label(self):
        return self._label


class Core:
    def __init__(self, clock: float, cache: float, bogomips: float, label: str):
        self._label: str = label
        self._bogomips: float = bogomips
        self._cache: float = cache
        self._clock: float = clock

    def get_bogomips(self):
        return self._bogomips


class Test:
    def __init__(self, duration: str or float, job_id: None or str, name: str, provides: str,
                 result: str, tags: str,
                 time: int):
        self._duration: float = float(duration)
        self._id: None or str = job_id
        self._name: str = name
        self._provides: str = provides
        self._result: str = result
        self._tags: str = tags
        self._time: int = time

    def get_id(self):
        return self._id

    def get_time(self):
        return self._time

    def get_duration(self):
        return self._duration


class JobMetadata:
    def __init__(self, job_metadata_dict=None):
        self._tests = []
        self._id = None
        self._nodes = []
        if job_metadata_dict is not None:
            self.init_from_pcp_output(job_metadata_dict)

    def filter_tests(self, test_list: List[str]):
        test_list = list(map(lambda x: x.strip(), test_list))
        self._tests = list(filter(lambda x: x.id is not None and x.id.strip() in test_list,
                                  self._tests))

    def init_from_pcp_output(self, job_metadata_dict: Dict):
        self._tests: List[Test] = [
            Test(metadata["duration"], metadata["id"], metadata["name"], metadata["provides"],
                 metadata["result"],
                 metadata["tags"], metadata["time"]) for metadata in
            job_metadata_dict["metadata"]["tests"]]
        self._id: int = job_metadata_dict["id"]
        self._nodes: List[NodeMetadata] = [NodeMetadata(node_metadata_dict) for node_metadata_dict
                                           in
                                           job_metadata_dict["nodes"]]

    def get_nodes(self):
        return self._nodes

    def get_tests(self):
        return self._tests

    def get_id(self):
        return self._id


class NodeMetadata:
    def __init__(self, node_metadata_dict: dict):
        self._name: str = node_metadata_dict["name"]
        self._cores: List[Core] = [
            Core(core["clock"], core["cache"], core["bogomips"], core["label"]) for core in
            node_metadata_dict["metadata"]["cores"]]
        self._physical_memory: float = node_metadata_dict["metadata"]["physical_memory"]
        self._plots: List[Plot] = []
        self._valid_data = {}
        for plot in node_metadata_dict["graph"]:
            try:
                self._plots.append(Plot(plot["label"], plot["data"]))
                self._valid_data[plot["label"]] = True
            except NoDataError as e:
                self._valid_data[plot["label"]] = False
                print(e)

    def get_cores(self):
        return self._cores

    def get_physical_memory(self):
        return self._physical_memory


def delete_row_with_id(path: str, job_id: int, index: int) -> None:
    """
    Delete row with JOB_ID, so data can be updated or just deleted.
    :param path: Path to file
    :param job_id: Id of job that we want to delete data
    :param index: Index of JOB id in file.
    """
    if not os.path.exists(path):
        return
    file: TextIO = open(path, "r")
    rows: List[str] = list(
        filter(lambda x: x.split(",")[index].strip() != str(job_id), file.readlines()))
    file.close()
    file = open(path, "w")
    for row in rows:
        file.write(row)
    file.close()


def get_core_multiplier(number_of_cores: int) -> float:
    """
    Additional cores have some "penalization" for their power, this penalization is counted
    in this function. 3 cores wont count as 3xCPU_POWER but only 2,5xCPU_POWER etc.
    :param number_of_cores: Number of cores from node
    :return: Return core multiplier which is used to edit data in normalization.
    """
    base_value: int = 1
    multiplier: int = 1
    for i in range(1, number_of_cores):
        base_value += multiplier
        multiplier /= 2
    return base_value


def create_directories_recursive(path: str) -> None:
    """
    Gets path and creates directories in this path recursively.
    """
    path_directories: List[str] = path.split("/")
    path: str = ""
    for index in range(len(path_directories)):
        path = path + path_directories[index] + "/"
        if not os.path.isdir(path):
            os.mkdir(path)


def edit_memory_values_by_hardware(node_metadata: NodeMetadata, value: float):
    """
     Edits data by some BASIC_MEM_VALUE so jobs with different HW specs are normalized to same format.
    :param node_metadata: Node metadata with HW info.
    :param value: Value of MEM_AVAILABLE metric
    :return: Recounted MEM_AVAILABLE value by HW specs
    """
    return (ConstantValues.MEM_VALUE.value - node_metadata.get_physical_memory() * (
            1 - value)) / ConstantValues.MEM_VALUE.value


def parse_json_parameters(path: str) -> dict or list:
    """
    It gets path which is path to some file with JSON object. Then reads it and returns JSON object as a dict.
    :param path: Path to file with JSON object.
    :return: JSON object as Python data structure.
    """
    with open(path, 'r') as file:
        data = file.read().replace('\n', '')
        data = data.strip()
    return json.loads(data)


def interpolate_to_different_timeframe(array_with_values: List[float or None], timeframe: int) -> \
        List[float]:
    """
    ML needs dataset in same format, so different duration of tests have to be recounted to
    same time format. This function gets list of some length and recount it to length of timeframe
    parameter.
    :param array_with_values: Array which will be recounted to timeframe length
    :param timeframe: Len of the output
    :return: List of length timeframe with recounted data
    """
    result_array: List[float] = []
    converting_value: float = (len(array_with_values) - 1) / (timeframe - 1)
    actual_value: float = 0
    index: int = 0
    while index < len(array_with_values):
        if actual_value == index:
            result_array.append(array_with_values[index])
            actual_value += converting_value
        elif actual_value > index:
            index += 1
        else:
            result_array.append(
                (array_with_values[index] - array_with_values[index - 1]) * (
                        actual_value - index + 1) + (
                    array_with_values[index - 1]))
            actual_value = actual_value + converting_value
    if len(result_array) != timeframe:
        result_array.append(array_with_values[-1])
    return result_array


def parse_single_charts_from_chart_set(string_with_values: str) -> Dict[str, List[float or None]]:
    """
    Get string with metrics and values, extracts them and return them as dict. Example:
    'Time X Y\n1 0.2 0.1\n2 0.4 None' -> {'Time':[1,2],'X':[0.2,0.4],'Y':[0.1,None]}
    :param string_with_values: String which contains the values
    :return: Dict with time and values in format as in example.
    """
    dict_with_charts: Dict[str, List[float or None]] = {}
    parsed_string_to_rows: List[str] = string_with_values.strip().split("\n")[:-1]
    labels: List[str] = parsed_string_to_rows[0].split(" ")
    label_values: List[List[float or None]] = [[] for _ in range(len(labels))]
    for row_index in range(1, len(parsed_string_to_rows)):
        values: List[str] = parsed_string_to_rows[row_index].split(" ")
        for value_index in range(len(values)):
            try:
                label_values[value_index].append(float(values[value_index]))
            except ValueError:
                label_values[value_index].append(None)
    for label_index in range(len(labels)):
        dict_with_charts[labels[label_index]] = label_values[label_index]
    return dict_with_charts


def clean_output_from_variable_plots(pcp_output: dict) -> None:
    """
    Clean input with job information from plots with variable length of columns and value datasets.
    :param pcp_output: JSON with beaker job information as dict, from which are extracted variable length plots.
    """
    for node in pcp_output["nodes"]:
        node["graph"] = [plot for plot in node["graph"] if plot["label"] == PlotNames.CPUS.value]


def extract_mem_ava_to_single_plot(pcp_output: dict) -> None:
    """
    Extract mem_ava columns from cpu plot to single plot.
    :param pcp_output: SON with beaker job information as dict, which is edited and added new plot with mem_ava values
    """
    for node in pcp_output["nodes"]:
        for plot in node["graph"]:
            if plot["label"] == PlotNames.CPUS.value:
                rows = plot["data"].split("\n")[:-1]
                index_of_mem = rows[0].split(" ").index(InputDataLabels.MEM_AVA.value)
                mem_ava_plot: List[str] = list(
                    map(lambda x: "{0} {1}\n".format(x.split(" ")[0], x.split(" ")[index_of_mem]),
                        rows))
                plot["data"] = "\n".join(
                    list(map(lambda x: " ".join(x.split(" ")[:index_of_mem]), rows)))
                node["graph"].append(
                    {"label": PlotNames.MEM_AVAILABLE.value, "data": "".join(mem_ava_plot)})


def normalize_none_values_from_array(value_list: List[float or None]) -> None:
    """
    Gets value_list which is normalized so it wont contain any None values.
    :param value_list: List in format for example [0.2,0.5,None,None]
    """
    list_with_nones: List[None] = list(filter(lambda x: x is None, value_list))
    if len(list_with_nones) / len(value_list) > 0.5:
        raise NoDataError  # when None are more than 50% of the dataset, there are corrupted
        # information

    value_before = 0
    none_count = 0
    for value_index in range(len(value_list)):
        if value_list[value_index] is None:
            none_count += 1
        else:
            for i in range(none_count):
                value_list[value_index - i - 1] = value_before + (
                        value_list[value_index] - value_before) * (
                                                          (none_count - i) / (none_count + 1))
            none_count = 0
            value_before = value_list[value_index]
    for i in range(none_count):
        value_list[len(value_list) - i - 1] = value_before + (0 - value_before) * (
                (none_count - i) / (none_count + 1))


def get_minimum_and_maximum(job_metadata: JobMetadata, label: str) -> (int, int):
    """
    Get minimum and maximum time of plots with label which is given as parameter
    :param job_metadata: Job metadata with all plots and data from job from beaker
    :param label: Label of plots, that we want minimum and maximum for
    :return: Return minimum and maximum in format (int,int)
    """
    minimum, maximum = math.inf, -math.inf
    for node in job_metadata.get_nodes():
        plot = next(x for x in node.plots if x.get_label() == label)
        minimum, maximum = min(minimum,
                               plot.get_value_by_label_and_index(InputDataLabels.TIME.value,
                                                                 0)), max(
            maximum, plot.get_value_by_label_and_index(InputDataLabels.TIME.value, -1))
    return int(minimum), int(maximum)


def output_memory_available_data(job_metadata: JobMetadata) -> Dict[str, List[float]]:
    """
    Get job_metadata, finds all MEM plots across nodes, edit data and output 1 normalize dataset, which is then saved in
    next function.
    :param job_metadata: Job metadata with all data about job from beaker
    :return: Data in format like ["Time":[1,2,3],"mem_available":[0.2,0.4,0.5] ...] (concrete keys are in settings.py)
    """
    minimum, maximum = get_minimum_and_maximum(job_metadata, PlotNames.MEM_AVAILABLE.value)  # get
    # minimum and maximum time from memory plot
    output_dict = {InputDataLabels.TIME.value: [], InputDataLabels.MEM_AVA.value: []}
    for time in range(minimum, maximum + 1):
        count = 0
        value = 0
        for node in job_metadata.get_nodes():
            plot = next(
                plot for plot in node.plots if plot.get_label() == PlotNames.MEM_AVAILABLE.value)
            time_index = plot.get_time(time)
            if time_index is not None:
                count += 1
                value += edit_memory_values_by_hardware(node,
                                                        plot.get_value_by_label_and_index(
                                                            InputDataLabels.MEM_AVA.value,
                                                            time_index))
        output_dict[InputDataLabels.TIME.value].append(time)
        output_dict[InputDataLabels.MEM_AVA.value].append(value / count)
    '''    
    for node in job_metadata.nodes:
        plot = next(plot for plot in node.plots if plot.label == PlotNames.MEM_AVAILABLE.value)
        print(plot.data["mem_ava_alert"][1157])
    print(output_dict["mem_ava_alert"][1157])'''
    return output_dict


def output_cpu_data(job_metadata: JobMetadata) -> Dict[str, List[float]]:
    """
    Get job_metadata, finds all CPU plots across nodes, edit data and output 1 normalize dataset, which is then saved in
    next function.
    :param job_metadata: Job metadata with all data about job from beaker
    :return: Data in format like ["Time":[1,2,3],"cpu_idle":[0.2,0.4,0.5] ...] (concrete keys are in settings.py)
    """
    minimum, maximum = get_minimum_and_maximum(job_metadata, PlotNames.CPUS.value)
    output_dict = {InputDataLabels.TIME.value: [], InputDataLabels.KSMD.value: [],
                   InputDataLabels.SYSTEM.value: [],
                   InputDataLabels.IDLE.value: [], InputDataLabels.USER.value: []}

    for node in job_metadata.get_nodes():
        plot = next(plot for plot in node.plots if plot.get_label() == PlotNames.CPUS.value)
        remove_steal_values(InputDataLabels.IDLE.value, InputDataLabels.STEAL.value, plot)
        plot.delete_dataset(InputDataLabels.STEAL.value)

    for time in range(minimum, maximum + 1):
        count = 0
        output_dict[InputDataLabels.TIME.value].append(time)
        temporary_dict = {InputDataLabels.KSMD.value: 0,
                          InputDataLabels.SYSTEM.value: 0,
                          InputDataLabels.IDLE.value: 0, InputDataLabels.USER.value: 0}
        for node in job_metadata.get_nodes():
            plot = next(plot for plot in node.plots if plot.get_label() == PlotNames.CPUS.value)
            time_index = plot.get_time(time)
            if time_index is not None:
                cpu_sum: float = 0
                cpu_base: float = plot.get_value_by_label_and_index(InputDataLabels.IDLE.value,
                                                                    time_index)
                count += 1
                for label in plot.get_data_labels():
                    if label != InputDataLabels.IDLE.value and label != InputDataLabels.TIME.value:
                        cpu_base += plot.get_value_by_label_and_index(label, time_index)
                        power_index = 1
                        plot.set_value_on_label_and_index(label, time_index,
                                                          power_index *
                                                          plot.get_value_by_label_and_index(
                                                              label,
                                                              time_index))
                        cpu_sum += plot.get_value_by_label_and_index(label, time_index)
                plot.set_value_on_label_and_index(InputDataLabels.IDLE.value, time_index,
                                                  cpu_base - cpu_sum)
                for key in plot.get_data_labels():
                    if key != InputDataLabels.TIME.value:
                        temporary_dict[key] += plot.get_value_by_label_and_index(key, time_index)
        for key in temporary_dict.keys():
            output_dict[key].append(temporary_dict[key] / count)

    return output_dict


def get_cpu_power_index(node_metadata: NodeMetadata) -> float:
    """
    Count CPU power index for node from parameters
    :param node_metadata: Node metadata which contains data about CPU cores
    :return: Index which is used to edit data about tests
    """
    final_index: float = 0
    multiplier: float = get_core_multiplier(len(node_metadata.get_cores()))
    for core in node_metadata.get_cores():
        final_index += core.get_bogomips()
    return (final_index / len(node_metadata.get_cores())) * multiplier


def remove_steal_values(idle_label: str, steal_label: str, plot: Plot) -> None:
    """
    This function removes values from steal metrics and add data to idle metric. Its important to have consistent data
    among multiple jobs from beaker. Steal time is caused by HW problems with virtual machines. If steal time occurs
    and idle time is <0.05 in same times, raises error, because data are corrupted by external effects.
    :param idle_label: Label with idle data.
    :param steal_label: Label with steal data
    :param plot: Plot in which are data edited
    """
    for index in range(len(plot.get_values_by_label(idle_label))):
        if plot.get_value_by_label_and_index(idle_label,
                                             index) - plot.get_value_by_label_and_index(
            steal_label, index) <= 0:
            print("Steal value is: {}".format(index))
            raise StealError
        else:
            plot.set_value_on_label_and_index(idle_label, index,
                                              plot.get_value_by_label_and_index(idle_label,
                                                                                index) + plot.get_value_by_label_and_index(
                                                  steal_label, index))


def get_parsed_data_by_test(final_output: FinalDataOutput, job_metadata: JobMetadata):
    """
    This function gets datasets for concrete pcp_output and concrete type of data('cpus' or
    'mem_available') and removes Time values, parse datasets by testcases and returns it as
    dictionary.
    :param final_output: FinalOutput structure with data for concrete type of datasets for
    concrete job.
    :param job_metadata: JobMetadata structure with information about pcp_output.
    :return: Parsed data by testcases. -> {'testcase_2' : {'mem_ava_alert':[0.58,0.59] },
                                           'testcase_3' : {'mem_ava_alert':[0.63,0.29] }
                                           }
    """
    result_output = {}
    for test in job_metadata.get_tests():
        if test.get_id() is not None:
            result_output[test.get_id()] = {}
            time_index = final_output.get_time(int(test.get_time()))
            duration = int(test.get_duration())
            for key in final_output.get_values():
                if key != InputDataLabels.TIME.value and duration > 5:
                    values = final_output.get_values_by_label(key)[time_index:time_index + duration]
                    interpolated_values = interpolate_to_different_timeframe(values, 1000)
                    result_output[test.get_id()][key] = interpolated_values
    return result_output


def write_data(final_output: FinalDataOutput, job_metadata: JobMetadata, output_dir: str):
    """
    Gets final_output with data to write. Creates needed directories, files and write data in some hierarchy.
    :param final_output: Output with data to be written.
    :param job_metadata: Metadata with all information about beaker job.
    :param output_dir: Root directory with data for ML.
    """
    path: str = Paths.NODE_DIR.value.format(output_dir, final_output.get_name(),
                                            len(job_metadata.get_nodes()))
    create_directories_recursive(path)

    for test in job_metadata.get_nodes():
        if test.get_id() is not None:
            time_index = final_output.get_time(int(test.get_time()))
            duration = int(test.get_duration())
            for key in final_output.get_values():
                create_directories_recursive("{}/{}".format(path, test.get_id()))
                if key != InputDataLabels.TIME.value and duration > 5:
                    values = final_output.get_values_by_label(key)[time_index:time_index + duration]

                    interpolated_values = interpolate_to_different_timeframe(values, 1000)
                    create_labels("{}/{}/{}".format(path, test.get_id(), key))
                    delete_row_with_id("{}/{}/{}".format(path, test.get_id(), key),
                                       job_metadata.get_id(), -2)

                    file = open("{}/{}/{}".format(path, test.get_id(), key), "a")
                    file.write(",".join(list(map(lambda x: str(x), interpolated_values))))
                    file.write(",{},{}\n".format(job_metadata.get_id(), final_output.get_result()))
                    file.close()


def delete_driver_from_input(input_json):
    """
    Deletes nodes which are marked as driver ones.
    :param input_json: Input json from pcp-log-analysis as dictionary.
    """
    for node_index in range(len(input_json["nodes"])):
        if input_json["nodes"][node_index]["driver"]:
            input_json["nodes"] = input_json["nodes"][:node_index] + input_json["nodes"][
                                                                     node_index + 1:]
            return


def check_if_valid(label: str, job_metadata: JobMetadata) -> bool:
    """
    Gets job_metadata and label, checks if job_metadata contains data with that label. Returns
    True if yes, else returns False
    :param label: Label of data that we want to check.
    :param job_metadata: JobMetadata structure.
    :return: True if job_metadata contains data with label, else False.
    """
    for node in job_metadata.get_nodes():
        if not node.valid_data[label]:
            return False
    return True


def create_labels(path: str):
    """
    This function creates file in path and writes labels. If file already exists, does nothing.
    :param path: Path to file that we want to create and create labels in.
    """
    if not os.path.exists(path):
        file = open(path, "w")
        for x in range(1, DataFormat.ROW_NUMBER.value + 1):
            file.write("{},".format(x))
        file.write("ID,Result\n")
        file.close()


def get_data_from_file(path: str, result: Result) -> pd.DataFrame:
    """
    This function gets path to file with datasets and result that we want to get. Then returns
    all datasets filtered by result. Column with result is dropped.
    :param path: Path to file with datasets
    :param result: Result of datasets that we want to get
    :return: Datasets without column Result
             -> [ [ 0.2, 0.01,10000], [ 0.2, 0.1, 10001 ] ]
    """
    dataset = pd.read_csv(path)
    has_value_as_param = dataset['Result'] == result
    dataset = dataset[has_value_as_param]
    dataset = dataset.drop('Result', axis=1)

    return dataset


def get_datasets_for_all_metrics(data_dir: str, ml_type: str, nodes: int, test_id: str):
    """
    This functions get root dir with datasets, type of dataset, number of nodes and testcase id.
    Then gets data from all metrics for concrete type of dataset and returns them in some predefined
    way as nested list.
    :param data_dir: Path to root dir with data and models for ML analysis.
    :param ml_type: Type of datasets -> 'cpus'
    :param nodes: Number of nodes -> 2
    :param test_id: TestCase id -> 'testcase_1'
    :return: Datasets for entered parameters
             -> [ [ [0.1, 0.2], [0.01, 0.9] ], [ [0.2, 0.4],[0.1, 0.7] ] ]
    """
    new_dataset = pd.DataFrame()  # we initiate new pd dataframe

    for metric in Order.ORDER.value[ml_type]:
        # we gets data from the single files for all dimensions(1 dimension is 1 metric),
        # we have to check, if is dataset empty so we can initiate new files or just merge with
        # existing ones, we join dimensions on ID
        if new_dataset.empty:
            new_dataset = get_data_from_file(
                Paths.DATA_FILE.value.format(data_dir, ml_type, nodes, test_id,
                                             metric), Result.CORRECT.value)
        else:
            new_dataset = new_dataset.merge(get_data_from_file(Paths.DATA_FILE.value.format(
                data_dir,
                ml_type,
                nodes, test_id,
                metric),
                Result.CORRECT.value),
                on="ID")
    list_value = list(new_dataset.values)  # we convert Dataframe to list
    # we gets datasets in concat format, so we need to connect dimensions and edit format of dataset
    # each point from the datasets is now represented by list instead of float, list has all point
    # dimension in format set in settings file
    for row_index in range(len(list_value)):
        new_row_value = []
        for column_index in range(int(len(list_value[row_index]) / len(Order.ORDER.value[
                                                                           ml_type]))):
            new_row_value.append([list_value[row_index][x] for x in range(column_index,
                                                                          len(list_value[
                                                                                  row_index]),
                                                                          1000)])
        list_value[row_index] = new_row_value
    return list_value


def count_deviation_for_point(x_list: List[float], y_list: List[float]) -> float:
    """
    This function gets 2 points and returns deviation between them. Points are multidimensional.
    :param x_list: First point -> [0.1, 0.2]
    :param y_list: Second point -> [0.1, 0.2]
    :return: Deviation -> 0
    """
    constant = 0
    for index in range(len(x_list)):
        constant += (x_list[index] - y_list[index]) ** 2
    return math.sqrt(constant)


def count_deviation_for_row(x_list: List[List[float]], y_list: List[List[float]]) -> float:
    """
    This function gets 2 datasets and returns deviation between them.
    :param x_list: First dataset -> [ [0.1, 0.2], [0.01, 0.9] ]
    :param y_list: Second dataset -> [ [0.1, 0.2], [0.01, 0.9] ],
    :return: Deviation between datasets -> 0
    """
    deviation_list = []
    for index in range(len(x_list)):
        deviation_list.append(count_deviation_for_point(x_list[index], y_list[index]))
    return sum(deviation_list) / len(deviation_list)


def count_deviation_for_datasets(x_list: List[List[List[float]]], y_list: List[List[List[float]]]) \
        -> List[float]:
    """
    This function gets 2 datasets and returns list of deviations between entry
    datasets.
    :param x_list: First list of datasets
    -> [ [ [0.1, 0.2], [0.01, 0.9] ], [ [0.2, 0.4],[0.1, 0.7] ] ]
    :param y_list: Second list of datasets
    -> [ [ [0.1, 0.2], [0.01, 0.9] ], [ [0.2, 0.4],[0.1, 0.7] ] ]
    :return: List of deviations, len is number of rows from datasets -> [0.05, 0.02]
    """
    deviation_list = []
    for index in range(len(x_list)):
        deviation_list.append(count_deviation_for_row(x_list[index], y_list[index]))
    return deviation_list


def get_id_from_beaker_results_file(path: str) -> int:
    """
    Return job id from beaker metadata file.
    :param path: Path to file which is in beaker results.json format.
    :return: Job ID from beaker metadata file.
    """
    json_object = parse_json_parameters(path)
    return int(json_object["metadata"]["job_id"])


def get_id_from_pcp_json(path: str) -> int:
    """
    Return job id from PCP output file.
    :param path: Path to file which is in PCP JSON format.
    :return: Job ID from PCP output.
    """
    json_object = parse_json_parameters(path)
    return int(json_object["id"])


def get_count_of_nodes_from_pcp_json(path: str) -> int:
    """
    Return number of nodes from PCP output file.
    :param path: Path to file which is in PCP JSON format.
    :return: Number of nodes in PCP output.
    """
    json_object = parse_json_parameters(path)
    return int(len(json_object["nodes"]) - 1)


def check_if_file_is_accessible(path: str, mode: str) -> bool:
    """
    Check if is file accessible in concrete mode.
    :param path: Path to file
    :param mode: Open mode
    :return: True if accessible, else False
    """
    try:
        file = open(path, mode)
        file.close()
    except Exception as e:
        logger.error(e)
        return False
    return True


def get_datasets_for_types(path: str, test_list: List[str], skip_cpus: bool,
                           skip_memory: bool, job_metadata: JobMetadata,
                           result: Result =
                           Result.CORRECT.value) -> List[FinalDataOutput]:
    """
    This function provides preparation for saving of data or executing ML analysis. It gets path
    to file and other parameters and return datasets as list of FinalDataOutput structures.
    :param path: Path to dataset -> './output20500.json'
    :param test_list: List with testcases, which we want to save -> ['testcase_1','testcase_2']
    :param job_metadata: JobMetadata structure with information about entered job dataset.
    :param skip_cpus: Boolean value if we want to skip cpus type of datasets
    :param skip_memory:  Boolean value if we want to skip memory type of datasets
    :param result: Result if dataset is valid or not
    :return:
    """
    datasets = []
    pcp_output = parse_json_parameters(path)  # return whole element of input

    delete_driver_from_input(pcp_output)  # delete driver node from input
    clean_output_from_variable_plots(
        pcp_output)  # clean code from plots with disks, which have variable index of columns
    extract_mem_ava_to_single_plot(
        pcp_output)  # extracts mem_ava value from CPUS plot set and creates new set

    job_metadata.init_from_pcp_output(pcp_output)
    if test_list:
        job_metadata.filter_tests(test_list)
    if check_if_valid(PlotNames.MEM_AVAILABLE.value, job_metadata) and not skip_memory:
        final_memory_output = FinalDataOutput(PlotNames.MEM_AVAILABLE.value,
                                              output_memory_available_data(job_metadata), result)
        datasets.append(final_memory_output)
    if check_if_valid(PlotNames.CPUS.value, job_metadata) and not skip_cpus:
        try:
            final_cpu_output = FinalDataOutput(PlotNames.CPUS.value, output_cpu_data(job_metadata),
                                               result)
            datasets.append(final_cpu_output)
        except StealError as e:
            print(e)
    return datasets


def save_data_for_single_output(path: str, test_list: List[str], output_directory: str, skip_cpus: bool,
                                skip_memory: bool,
                                result: Result =
                        Result.CORRECT.value):
    """
    This function gets data for concrete PCP output and saves them by defined format in settings to
    directory.
    :param path: Path to dataset -> './output20500.json'
    :param test_list: List with testcases, which we want to save -> ['testcase_1','testcase_2']
    :param output_directory: Root output directory for data -> './data'
    :param skip_cpus: Boolean value if we want to skip cpus type of datasets
    :param skip_memory:  Boolean value if we want to skip memory type of datasets
    :param result: Result if dataset is valid or not
    """
    job_metadata = JobMetadata()
    list_with_datasets = get_datasets_for_types(path, test_list, skip_cpus,
                                                skip_memory, job_metadata,
                                                result)
    for dataset in list_with_datasets:
        write_data(dataset, job_metadata, output_directory)
