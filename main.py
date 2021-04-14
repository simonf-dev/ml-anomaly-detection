import base64
import logging
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from ml_analysis_library import check_by_model, MLResultOutput, get_data_for_single_output
from plotting_library import parse_picture
from models_training import retrain_models_for_ml_type
from option_parsers import ml_entry_option_parser, save_option_parser, retrain_option_parser, \
    analysis_option_parser, print_main_help, get_plot_parser, get_logger
from pcplog.pcpanalysis import make_tests_cqe
from settings import LoggingSettings, PlotNames
from parsing_library import parse_json_parameters, \
    check_if_file_is_accessible, get_id_from_beaker_results_file, get_id_from_pcp_json, \
    get_count_of_nodes_from_pcp_json, save_data_for_single_output

logger = get_logger(LoggingSettings.LOGGING_LEVEL.value, LoggingSettings.OUTPUT_FILE.value, "main")


def execute_pcp_analysis() -> bool:
    logger.debug("Starting to execute PCP analysis.")
    parser = analysis_option_parser()
    (options, args) = parser.parse_args()  # parse args from input
    if not options.archive or not options.metadata or not options.xml:
        # check if user entered all needed opts
        logger.error("You didn't specify correctly --archive, --metadata and --xml-metadata"
                     "options, which are compulsory.")
        return False

    # converts paths to absolute ones
    archive_path = os.path.abspath(options.archive)
    metadata_path = os.path.abspath(options.metadata)
    xml_path = os.path.abspath(options.xml)
    # if its something wrong with files, whole logic ends
    if not check_if_file_is_accessible(archive_path, "r") or not check_if_file_is_accessible(
            metadata_path, "r") or not check_if_file_is_accessible(xml_path, "r"):
        logger.error("Problem with accessing files from input from options --archive and "
                     "--metadata --xml-metadata.")
        return False
    # we get job_id from metadata file
    job_id = get_id_from_beaker_results_file(options.metadata)
    try:
        pcp_output = make_tests_cqe(job_id, archive_path, metadata_path, xml_path)
    except Exception as e:
        logger.error("Problem occurred in pcp analysis for job with id {}. Error message:"
                     .format(job_id, e))
        return False
    if options.output_file:  # if output_file not defined, we send result to stdout
        if not check_if_file_is_accessible(options.output_file, "w"):
            # if there is problem with output file, we ends function
            logger.error("Problem with accessing files from input from options --output-file: {}"
                         .format(options.output_file))
            return False
        sys.stdout = open(options.output_file, "w")
    print(pcp_output)
    logger.debug("Ending PCP analysis.")
    return True


def execute_save_of_data() -> bool:
    logger.setLevel(logging.DEBUG)
    logger.debug("Starting saving data for ML.")
    parser = save_option_parser()
    (options, args) = parser.parse_args()
    if not options.input_file:
        logger.error("Input JSON file with data not specified, so nothing to do.")
        return False
    if not check_if_file_is_accessible(options.input_file, "r"):
        logger.error("Problem occurred with reading of input file.")
        return False
    if not options.output_dir:
        logger.error("Root output directory for data not specified, so we cannot write output "
                     "data.")
        return False
    test_list = []
    if options.test_list != "":
        test_list = options.test_list.split("&")
    input_file = os.path.abspath(options.input_file)
    output_dir = os.path.abspath(options.output_dir)
    save_data_for_single_output(input_file, test_list, output_dir, options.skip_cpus,
                        options.skip_memory)
    logger.debug("Ending saving data for ML.")
    return True


def execute_ml_for_entry() -> bool:
    logger.debug("Starting to get ML result for the entry. Command is: {}".format(" ".join(
        sys.argv)))
    parser = ml_entry_option_parser()
    (options, args) = parser.parse_args()
    if not options.input_file:
        logger.error("Input JSON file with data not specified, so nothing to do.")
        return False
    input_file = os.path.abspath(options.input_file)
    if not check_if_file_is_accessible(input_file, "r"):
        logger.error("Problem occurred with reading of input file.")
        return False
    if not options.data_directory:
        logger.error("Data directory not specified, so nothing to do.")
        return False
    test_list = []
    if options.test_list != "":
        test_list = options.test_list.split("&")
    input_file = os.path.abspath(options.input_file)
    dataset = get_data_for_single_output(input_file, test_list, options.skip_cpus,
                              options.skip_memory)  # we get data frm pcp_file in correct format
    ml_metadata = MLResultOutput(get_id_from_pcp_json(input_file))  # we create output structure
    check_by_model(dataset, options.data_directory, get_count_of_nodes_from_pcp_json(input_file),
                   ml_metadata)  # we execute ML analysis for concrete dataset
    if options.logical_output:
        print(ml_metadata.as_json())
        return True
    print(ml_metadata.as_nice_string())
    logger.debug("Getting  ML result for the entry successfully ended")
    return True


def execute_retrain_of_models() -> bool:
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
    retrain_successful = True
    if not options.skip_cpus:
        retrain_successful = retrain_models_for_ml_type(data_dir, PlotNames.CPUS.value, nodes,
                                                        test_list)
    if not options.skip_memory:
        return retrain_successful and retrain_models_for_ml_type(data_dir,
                                                                 PlotNames.MEM_AVAILABLE.value,
                                                                 nodes, test_list)


def execute_picture_logic() -> bool:
    """
    Function to parse picture with defined opts.
    :return: True if everything went ok, else false.
    """
    logger.debug("Starting to getting picture.")
    parser = get_plot_parser()
    (options, args) = parser.parse_args()
    if not options.input_file:
        logger.error("Missing input file option, nothing to do.")
        return False
    input_file = os.path.abspath(options.input_file)
    pcp_output = parse_json_parameters(input_file)
    picture, err = parse_picture(pcp_output)
    if options.output_file:
        output_file = os.path.abspath(options.output_file)
        output = open(output_file, "wb")
        output.write(base64.b64decode(picture))
        output.close()
        return True
    print(base64.b64decode(picture))
    logger.debug("Successfully ending getting picture.")
    return True


if __name__ == '__main__':
    print(os.getcwd())
    if sys.argv[1] == "make-analysis":
        return_value = execute_pcp_analysis()
    elif sys.argv[1] == "save-result":
        return_value = execute_save_of_data()
    elif sys.argv[1] == "retrain-model":
        return_value = execute_retrain_of_models()
    elif sys.argv[1] == "get-result":
        return_value = execute_ml_for_entry()
    elif sys.argv[1] == "get-picture":
        return_value = execute_picture_logic()
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
