import base64
import os
import random
import shutil
import string
import subprocess
import tempfile

from option_parsers import get_logger
from settings import LoggingSettings, Paths

logger = get_logger(LoggingSettings.LOGGING_LEVEL.value, LoggingSettings.OUTPUT_FILE.value,
                    "plotting_library")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_random_string(length):
    """
    Generates random string in length == length from lowercase letters.
    """
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def NoDataError(BaseException):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        if self.message:
            return self.message
        return "NoDataError has been raised."


def get_picture(pcp_json_output, tmpdir):
    """
    This function is responsible for whole image logic. It gets job_id and tmpdir(where temp files should be saved).
    It computes or gets from database all important informations, call gnuplot and return its output.
    @param beaker_id: Job id from the testrun.
    @param tmpdir: Path to tmpdir.
    """
    job_id = pcp_json_output["id"]
    plots = []
    for node in pcp_json_output["nodes"]:
        for plot in node["graph"]:
            plots.append(plot)

    if len(plots) == 0:
        logger.error(
            "Error in parsing PCP log plots: No plots found in database for job with id {}".format(
                job_id))
        raise NoDataError("No plots found in database for job with id {}".format(
            job_id))  # if they are no plots, something is wrong and we raise error
    # we check what version of pcp_output is for these concrete plots
    version = pcp_json_output["output_version"]
    datatypes = version_datatypes(
        version)  # we get plot labels and their order for concrete pcp_output versions
    hosts = get_node_names(pcp_json_output)  # we get all node names for concrete plots
    max_y = get_max_y(datatypes,
                      plots)  # gives us list of maximumf y axis for concrete type of plots
    plot_files, threshold_files = make_plot_files(plots, hosts, datatypes,
                                                  tmpdir)  # creates us files with plot data and threshold data
    print(os.getcwd())

    gnuplot_command = ["gnuplot",
                       "-e", "tmpdir='{}'".format(tmpdir),
                       "-e", "plot_files='{}'".format(" ".join(plot_files)),
                       "-e", "threshold_files='{}'".format(" ".join(threshold_files)),
                       "-e", "beaker_job_id='J:{}'".format(job_id),
                       "-e", "node_hostnames='{}'".format(" ".join(hosts)),
                       "-e", "data_types='{}'".format(" ".join(datatypes)),
                       "-e", "max_y='{}'".format(" ".join(max_y)),
                       Paths.GNUPLOT_SCRIPT_PATH.value.format(version), ]
    logger.debug("Executing gnuplot command: {}".format(" ".join(gnuplot_command)))
    proc = subprocess.Popen(
        gnuplot_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = proc.communicate()
    return out, err


def version_datatypes(version):
    """
    Function which identify order and name of plots in PCP log output by version number.
    When comes new version of pcp-log analysis, we will add new set of datatypes to synchronize it with it.
    Every pcpanalysis for current job keeps his version in the database. It helps us to keep this piece of code
    compatible with new versions of PcpAnalysis tool. Version datatypes are also synchronize with gnuplot
    script, so we know index of each type of plot and we have written script by it.
    @param version: Version of plots.
    @return: List of name of plots.
    """
    return [["storage_percent", "storage_size", "cpus", "units"],
            ][version]


def get_max_y(datatypes, plots):
    """
    Gives max y axis value for each category of plots, so each category has same scaling for all nodes.
    It helps with better clarity of plots.
    datatypes = ["cpu"]
    plots = [Plot {node.name = "a",label="cpu",max_y=20, ...}, Plot {node.name = "b",label="cpu",max_y=50,...}
    -> [50]
    @param plots: List of Plot database entities.
    @param datatypes: List of type of plots as strings. We get it from function version_datatypes.
    @return: List of max_y for each category of plots.
    """
    max_y = []
    for type in datatypes:
        for plot in plots:
            if plot["label"] == type:
                max_y.append(str(plot["max_y"]))
                break
    return max_y


def get_time_extremes(data):
    """
    Gives the smallest and the highest time of plots.
    It helps with better clarity of plots.
    'Time a b\n 1 2 4\n2 3 4\n3 2 1' -> '1','3'

    @param data: Dataset for concrete plot
    @return: Minimum of time value for the dataset, Maximum of time value for the dataset
    """

    data = data.split('\n')
    index = data[0].index("Time")
    return data[1][index], data[-1][index]


def get_node_names(pcp_json_output):
    """
    Function to get names of nodes.
    @param pcp_json_output: PCP json output as dictionary.
    @return: List of names of all nodes as strings.
    """
    hosts = []
    for node in pcp_json_output["nodes"]:
        if node["name"] not in hosts:
            hosts.append(str(node["name"]).strip())
    return hosts


def make_plot_files(plots, nodes, datatypes, tmpdir):
    """
    Function get all plots for current job, list of nodes and list of datatypes of nodes.
    For example : plots = list of 4 Plot objects from DB,  nodes: List[str] = ["A","B"], datatypes = ["a","b"].
    Function connects plot with node and datatype and saves them to file. Then return file names in format
    ["file_for_Aa","file_for_Ab","file_for_Ba","file_for_Bb"]. Same output logic for threshold_files.
    It saves all files to dir with tmpdir path.
    @param plots: List of Plot objects from database.
    @param nodes: List with names of all nodes from current job.
    @param datatypes: List with names of plots, its compatible with pcp_output from concrete version.
    @param tmpdir: Path to tmpdir.
    @return: Return list with plot file names and list with threshold file names. Logic of output is in description.

    """
    logger.debug("Creating plot_files in dir {}".format(tmpdir))
    plot_files = []
    threshold_files = []
    for node in nodes:
        for datatype in datatypes:
            plot = next(plot for plot in plots if
                        plot["label"] == datatype)  # we got plot with correct label from the order
            plot_name, threshold_name = get_random_string(8), get_random_string(
                8)  # we got name for file_name and threshold_name
            file = open("{}/{}.dat".format(tmpdir, plot_name),
                        "w")  # if some open or write fails, all function fails, because we wont get files in good format
            file.write(plot["data"])
            file.close()
            file = open('{}/{}.dat'.format(tmpdir, threshold_name), "w")
            file.write(plot["thresholds"])
            file.close()
            plot_files.append("{}.dat".format(plot_name))  # we append plot_file name
            threshold_files.append("{}.dat".format(threshold_name))  # we append threshold_file name
    logger.debug("Creating plot_files in dir {} done.".format(tmpdir))
    return plot_files, threshold_files


def parse_picture(pcp_json_output):
    """
    Function which gets beaker_id, extracts all information from PCP log models and use
    Gnuplot to produce encoded XML SVG format of picture, which sends to frontend. Its
    called from clusterqe/views.py
    @param pcp_json_output: PCP result JSON structure as dictionary.
    @return: Return tuple (output,error), when Gnuplot command isnt succesfull and output is empty, then output string is empty too."
    """
    logger.debug("Starting parsing plot pictures for job id {}".format(pcp_json_output["id"]))
    tmpdir = tempfile.mkdtemp()
    try:
        out, err = get_picture(pcp_json_output,
                               tmpdir)  # this block is responsible for all image logic
    except Exception as error:  # we dont care which exception occurs, we want to only delete tmpdir and raise it upper again
        raise error
    finally:  # this block is always executed, so we are sure that temp dir will be deleted, but exception will go up
        shutil.rmtree(tmpdir)
    if out == "":
        logger.error(
            "Gnuplot command for job id {} successfully ended without fatal error, but output is empty.Error for gnuplot is {}".format(
                pcp_json_output["id"], err))
        return (out, err)
    out = base64.b64encode(out)
    logger.debug("Gnuplot command for job id {} successfully ended.".format(pcp_json_output["id"]))
    return ('{}'.format(out.decode()), err)
