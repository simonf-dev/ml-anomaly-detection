import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from typing import List, Dict
from models_training import retrain_models_for_ml_type
from option_parsers import ml_entry_option_parser, save_option_parser, retrain_option_parser, \
    analysis_option_parser, print_main_help
from pcplog.pcpanalysis import make_tests_cqe
from tensorflow import keras

from parsing_library import parse_json_parameters, clean_output_from_variable_plots, \
    extract_mem_ava_to_single_plot, \
    FinalDataOutput, JobMetadata, output_memory_available_data, output_cpu_data, write_data, \
    delete_driver_from_input, StealError, check_if_valid, get_parsed_data_by_test, ML_metadata, \
     check_if_file_is_accessible
from settings import PlotNames, Result, DataFormat, Order, Paths








def count_deviation(input_data, output_data):
    deviation_output = []
    for sample_index in range(len(input_data)):
        deviation = 0
        for dimension_index in range(len(input_data[sample_index])):
            deviation += (abs(input_data[sample_index][dimension_index] -
                              output_data[sample_index][
                                  dimension_index]))
        deviation_output.append(deviation / len(input_data[sample_index]))
    return sum(deviation_output) / len(deviation_output)


def get_id_from_beaker_results_file(path: str) -> int:
    json_object = parse_json_parameters(path)
    return int(json_object["metadata"]["job_id"])


def get_id_from_pcp_json(path: str) -> int:
    json_object = parse_json_parameters(path)
    return int(json_object["id"])


def get_count_of_nodes_from_pcp_json(path: str) -> int:
    json_object = parse_json_parameters(path)
    return int(len(json_object["nodes"]) - 1)


def get_data_for_ml(path: str, test_list: List[str], skip_cpus: bool,
                    skip_memory: bool,
                    result: Result =
                    Result.CORRECT.value):
    return_output = {}
    pcp_output = parse_json_parameters(path)  # return whole element of input

    delete_driver_from_input(pcp_output)  # delete driver node from input
    clean_output_from_variable_plots(
        pcp_output)  # clean code from plots with disks, which have variable index of columns
    extract_mem_ava_to_single_plot(
        pcp_output)  # extracts mem_ava value from CPUS plot set and creates new set

    job_metadata = JobMetadata(pcp_output)
    if test_list:
        job_metadata.filter_tests(test_list)
    if check_if_valid(PlotNames.MEM_AVAILABLE.value, job_metadata) and not skip_memory:
        final_memory_output = FinalDataOutput(PlotNames.MEM_AVAILABLE.value,
                                              output_memory_available_data(job_metadata), result)
        return_output[PlotNames.MEM_AVAILABLE.value] = get_parsed_data_by_test(final_memory_output,
                                                                               job_metadata)
    if check_if_valid(PlotNames.CPUS.value, job_metadata) and not skip_cpus:
        try:
            final_cpu_output = FinalDataOutput(PlotNames.CPUS.value, output_cpu_data(job_metadata),
                                               result)
            return_output[PlotNames.CPUS.value] = get_parsed_data_by_test(
                final_cpu_output,
                job_metadata)
        except StealError as e:
            print(e)
    return return_output


def parse_single_output(path: str, test_list: List[str], output_directory: str, skip_cpus: bool,
                        skip_memory: bool,
                        result: Result =
                        Result.CORRECT.value):
    pcp_output = parse_json_parameters(path)  # return whole element of input

    delete_driver_from_input(pcp_output)  # delete driver node from input
    clean_output_from_variable_plots(
        pcp_output)  # clean code from plots with disks, which have variable index of columns
    extract_mem_ava_to_single_plot(
        pcp_output)  # extracts mem_ava value from CPUS plot set and creates new set

    job_metadata = JobMetadata(pcp_output)
    if test_list:
        job_metadata.filter_tests(test_list)
    if check_if_valid(PlotNames.MEM_AVAILABLE.value, job_metadata) and not skip_memory:
        final_memory_output = FinalDataOutput(PlotNames.MEM_AVAILABLE.value,
                                              output_memory_available_data(job_metadata), result)
        write_data(final_memory_output, job_metadata, output_directory)
    if check_if_valid(PlotNames.CPUS.value, job_metadata) and not skip_cpus:
        try:
            final_cpu_output = FinalDataOutput(PlotNames.CPUS.value, output_cpu_data(job_metadata),
                                               result)
            write_data(final_cpu_output, job_metadata, output_directory)
        except StealError as e:
            print(e)


def execute_pcp_analysis():
    logger.debug("Starting to execute PCP analysis.")
    parser = analysis_option_parser()
    (options, args) = parser.parse_args()  # parse args from input
    if not options.archive or not options.metadata or not options.xml:
        # check if user entered all needed opts
        logger.error("You didnt specify correctly --archive, --metadata and --xml-metadata"
                     "options, which are compulsory.")
        sys.exit(1)

    # converts paths to absolute ones
    archive_path = os.path.abspath(options.archive)
    metadata_path = os.path.abspath(options.metadata)
    xml_path = os.path.abspath(options.xml)
    # if its something wrong with files, whole logic ends
    if not check_if_file_is_accessible(archive_path, "r") or not check_if_file_is_accessible(
            metadata_path, "r") or not check_if_file_is_accessible(xml_path, "r"):
        logger.error("Problem with accessing files from input from options --archive and "
                     "--metadata --xml-metadata.")
        sys.exit(1)
    # we get job_id from metadata file
    job_id = get_id_from_beaker_results_file(options.metadata)
    try:
        pcp_output = make_tests_cqe(job_id, archive_path, metadata_path, xml_path)
    except Exception as e:
        logger.error("Problem occured in pcp analysis for job with id {}. Error message:"
                     .format(job_id,e))
        return False
    if options.output_file:  # if output_file not defined, we send result to stdout
        if not check_if_file_is_accessible(options.output_file, "w"):
            # if there is problem with output file, we ends function
            logger.error("Problem with accessing files from input from options --output-file: {}"
                         .format(options.output_file))
            sys.exit(1)
        sys.stdout = open(options.output_file,"w")
    print(pcp_output)
    sys.exit(0)



def get_multidimension_data(test_dataset, order):
    multidimensional_array = []
    for index in range(DataFormat.ROW_NUMBER.value):
        time_sample = []
        for key in order:
            time_sample.append(test_dataset[key][index])
        multidimensional_array.append(time_sample)
    return multidimensional_array


def execute_ml_for_single_input(model_path: str, threshold_path: str, input_data: List) -> \
        (bool, float, float):
    model = keras.models.load_model(model_path)
    threshold = float(open(threshold_path).readline())
    predicted_data = model.predict([input_data])
    deviation = count_deviation(input_data, predicted_data[0])
    if deviation > threshold:
        return False, deviation, threshold
    return True, deviation, threshold


def check_by_model(datasets: Dict, data_dir: str, nodes: int, result_metadata: ML_metadata):
    for ml_type in datasets.keys():
        for test in datasets[ml_type].keys():
            model_path = Paths.MODEL_PATH.value.format(data_dir, ml_type, nodes, test)
            threshold_path = Paths.THRESHOLD_PATH.value.format(data_dir, ml_type, nodes, test)
            try:
                input_data = get_multidimension_data(datasets[ml_type][test],
                                                     Order.order.value[ml_type])
                is_valid, deviation, threshold = execute_ml_for_single_input(model_path,
                                                                             threshold_path,
                                                                             input_data)
            except Exception as e:
                logger.error("Loading ML and threshold for test id {} type of {} failed because of "
                             "error: {}"
                             .format(test, ml_type, e))
                result_metadata.add_result(test, ml_type)
                # we record that we tried to make analysis, but problem occured with ML
                continue
            # we log results to metadata_object
            result_metadata.add_result(test, ml_type, True, is_valid, threshold, deviation)


def save_data_for_ml():
    logger.setLevel(logging.DEBUG)
    print("Starting saving data for ML.")
    parser = save_option_parser()
    (options, args) = parser.parse_args()
    if not options.input_file:
        logger.error("Input JSON file with data not specified, so nothing to do.")
        sys.exit(1)
    if not check_if_file_is_accessible(options.input_file, "r"):
        logger.error("Problem occured with reading of input file.")
        sys.exit(1)
    if not options.output_dir:
        logger.error("Root output directory for data not specified, so we cannot write output "
                     "data.")
        sys.exit(1)
    test_list = []
    if options.test_list != "":
        test_list = options.test_list.split("&")
    input_file = os.path.abspath(options.input_file)
    output_dir = os.path.abspath(options.output_dir)
    parse_single_output(input_file, test_list, output_dir, options.skip_cpus,
                        options.skip_memory)
    print("Ending saving data for ML.")
    return True

def execute_ml_for_entry():
    parser = ml_entry_option_parser()
    (options, args) = parser.parse_args()
    if not options.input_file:
        logger.error("Input JSON file with data not specified, so nothing to do.")
        sys.exit(1)
    input_file = os.path.abspath(options.input_file)
    if not check_if_file_is_accessible(input_file, "r"):
        logger.error("Problem occured with reading of input file.")
        sys.exit(1)
    if not options.data_directory:
        logger.error("Data directory not specified, so nothing to do.")
        sys.exit(1)
    test_list = []
    if options.test_list != "":
        test_list = options.test_list.split("&")
    input_file = os.path.abspath(options.input_file)
    dataset = get_data_for_ml(input_file, test_list, options.skip_cpus,
                              options.skip_memory)
    ml_metadata = ML_metadata(get_id_from_pcp_json(input_file))
    check_by_model(dataset, options.data_directory, get_count_of_nodes_from_pcp_json(input_file),
                   ml_metadata)
    if options.logical_output:
        print(ml_metadata.as_json())
        sys.exit(0)
    print(ml_metadata.as_nice_string())
    sys.exit(0)





def retrain_models():
    """
    Function which parses all input parameters and retrain models by them.
    :return: If everything went ok, return True, else False.
    """
    parser = retrain_option_parser()
    (options, args) = parser.parse_args()
    if not options.data_dir:
        logger.error("Root directory with data not entered, nothing to do.")
        return False
    data_dir = os.path.abspath(options.data_dir)
    test_list = []
    if options.test_list != "":
        test_list = options.test_list.split("&")
    nodes = []
    if options.nodes != "":
        nodes = list(map(lambda x: int(x), options.nodes.split("&")))
    return_value = True
    if not options.skip_cpus:
        return_value = retrain_models_for_ml_type(data_dir, PlotNames.CPUS.value, nodes, test_list)
    if not options.skip_memory:
        return return_value and retrain_models_for_ml_type(data_dir, PlotNames.MEM_AVAILABLE.value,
                                                     nodes, test_list)


if __name__ == '__main__':

    if sys.argv[1] == "make-analysis":
        return_value = execute_pcp_analysis()
    elif sys.argv[1] == "save-result":
        return_value = save_data_for_ml()
    elif sys.argv[1] == "retrain-model":
        return_value = retrain_models()
    elif sys.argv[1] == "get-result":
        return_value = execute_ml_for_entry()
    elif sys.argv[1] == "-h" or sys.argv[1] == "--help":
        return_value = print_main_help()
    else:
        logger.error("Invalid first command, use option from: 'retrain-model','make-analysis"
                     ",'retrain-model','get-result'")
        sys.exit(1)
    if return_value:
        sys.exit(0)
    sys.exit(1)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
