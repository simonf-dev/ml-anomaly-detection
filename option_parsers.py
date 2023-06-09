import logging
import optparse
import os

file_handler = logging.FileHandler(filename='tmp.log')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def analysis_option_parser():
    parser = optparse.OptionParser(usage="main.py make-analysis [options]")
    parser.add_option("-a", "--archive", dest="archive", help="PCP zipped archive, from which "
                                                              "should be"
                                                              "analysis done")
    parser.add_option("-m", "--metadata", dest="metadata", help="Job JSON metadata file")
    parser.add_option("-x", "--xml-metadata", dest="xml", help="Job XML metadata file")
    parser.add_option("-o", "--output-file", dest="output_file", help="Path to output file, where"
                                                                      "will tool write output "
                                                                      "data, if missing, "
                                                                      "output will be printed to "
                                                                      "STDOUT")
    return parser


def ml_entry_option_parser():
    parser = optparse.OptionParser(usage="main.py ml-entry [options]")
    parser.add_option("-o", "--output-file", dest="output_file", help="Path to output file, where"
                                                                      "will tool write output "
                                                                      "data, if missing, "
                                                                      "output will be printed to "
                                                                      "STDOUT.")
    parser.add_option("-l", "--test-list", dest="test_list",
                      help="List of tests split by '&', because original tests id "
                           "contains ',' , so we cannot use ',' as delimiter here. When you "
                           "use this option, there will be saved only data for tests with "
                           "written ids. If its empty, all tests will be executed", default="")
    parser.add_option("-i", "--input-file", dest="input_file",
                      help="Path to JSON file created from pcp-analysis or from this tool with "
                           "command make-analysis.")
    parser.add_option("-d", "--data-directory", dest="data_directory",
                      help="Root directory with datasets for ML.")

    parser.add_option("--logical-output", dest="logical_output", action="store_true", default=False,
                      help="Choose if you want to get logical output instead of the nice one for"
                           "reading.")
    parser.add_option("--skip-cpus", dest="skip_cpus", action="store_true", default=False,
                      help="Choose if you want to skip generating cpus data from input file.")

    parser.add_option("--skip-memory", dest="skip_memory", action="store_true", default=False,
                      help="Choose if you want to skip generating memory data from input file.")
    return parser


def save_option_parser():
    parser = optparse.OptionParser(usage="main.py save-result [options]")
    parser.add_option("-l", "--test-list", dest="test_list",
                      help="List of tests split by '&', because original tests id "
                           "contains ',' , so we cannot use ',' as delimiter here. When you "
                           "use this option, there will be saved only data for tests with "
                           "written ids. If its empty, all tests will be executed", default="")

    parser.add_option("-o", "--output-directory", dest="output_dir",
                      help="Path to output directory, when we will write data. Tool have"
                           "fixed inner logic, so its best to create new empty directory in "
                           "the start specific for whole storing logic of this tool.")
    parser.add_option("-i", "--input-file", dest="input_file",
                      help="Path to JSON file created from pcp-analysis or from this tool with "
                           "command make-analysis.")
    parser.add_option("--skip-cpus", dest="skip_cpus", action="store_true", default=False,
                      help="Choose if you want to skip generating cpus data from input file.")

    parser.add_option("--skip-memory", dest="skip_memory", action="store_true", default=False,
                      help="Choose if you want to skip generating memory data from input file.")
    return parser


def retrain_option_parser():
    parser = optparse.OptionParser(usage="main.py save-result [options]")
    parser.add_option("-l", "--test-list", dest="test_list",
                      help="List of tests split by '&', because original tests id "
                           "contains ',' , so we cannot use ',' as delimiter here. When you "
                           "use this option, there will be saved only data for tests with "
                           "written ids. If its empty, all tests will be executed", default="")

    parser.add_option("-d", "--data-directory", dest="data_dir",
                      help="Path to output directory, when we will write data. Tool have"
                           "fixed inner logic, so its best to create new empty directory in "
                           "the start specific for whole storing logic of this tool.")
    parser.add_option("-n", "--nodes", dest="nodes", default="",
                      help="List of count of nodes, on which we want to retrain models. "
                           "Please split by &. If empty, then is retraining routine"
                           "executed on all nodes. Example:  2&4")

    parser.add_option("--skip-cpus", dest="skip_cpus", action="store_true", default=False,
                      help="Choose if you want to skip generating cpus data from input file.")

    parser.add_option("--skip-memory", dest="skip_memory", action="store_true", default=False,
                      help="Choose if you want to skip generating memory data from input file.")
    return parser


def get_plot_parser():
    parser = optparse.OptionParser(usage="main.py get-plot [options]")
    parser.add_option("-i", "--input-file", dest="input_file",
                      help="Path to JSON file created from pcp-analysis or from this tool with "
                           "command make-analysis.")
    parser.add_option("-o", "--output-file", dest="output_file",
                      help="Path to output file, when will tool save the picture. If not set,"
                           "it prints output to STDOUT.")
    return parser


def print_main_help():
    print("Available commands:\n\n"
          "make-analysis [opts] - executes PCP analysis and saves output to file or prints it to\n"
          "stdout\n\n"
          "save-result [opts] - parses *.json output from PCP analysis tool and saves it in"
          "format for ML\n\n"
          "get-result [opts] - parses *.json output and executes ML logic on it, then prints"
          "output to stdout with results\n\n"
          "retrain-model [opts] - retrains models and saves them\n\n"
          "get-plot [opts] - generates plot picture from PCP analysis output\n\n"
          "Write [COMMAND] --help to get info, how to use commands correctly")


def get_logger(log_level: logging, file_path, file_name):
    file_handler = logging.FileHandler(filename=file_path)
    logger = logging.getLogger(file_name)
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)
    return logger
