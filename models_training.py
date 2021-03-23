import logging
import os
import random
import re
from typing import List
import numpy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from neural_network_structure import get_neural_network
from parsing_library import get_dataset_for_all_metrics, count_deviation_for_dataset
from settings import Paths

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)



def retrain_model(data_dir: str, ml_type: str, nodes: int, test_id: str):
    """
    This function take parameters to identify for which type of data, test etc. should retrain
    ML model. Then retrains model and saves it. Also saves threshold, so we have all needed data
    to use the model in future.
    :param data_dir: Path to root data dir.
    :param ml_type: Type of components(cpus, memory)
    :param nodes: Number of nodes
    :param test_id: Id of testcase
    :return: True if everything went smoothly, in other cases False or throw Exception.
    """
    logger.debug(
        "Starting to retrain model for test_id: data_dir: {}, ml_type: {}, nodes: {}, "
        "test_id: {}.".format(data_dir, ml_type, nodes, test_id))
    neural_network = get_neural_network(ml_type)  # gets neural network for concrete ml_type
    input_data = get_dataset_for_all_metrics(data_dir, ml_type, nodes,
                                             test_id)  # get data in correct format for ML
    random.shuffle(input_data)  # shuffle data randomly
    train_data = numpy.array(input_data[int(len(input_data) / 10):])  # split data to train and test
    test_data = numpy.array(input_data[:int(len(input_data) / 10)])
    neural_network.fit(train_data, train_data, batch_size=150,
                       epochs=150,
                       validation_split=0.1,
                       verbose=1)
    output_data = neural_network.predict(test_data)  # predict data
    deviations = count_deviation_for_dataset(test_data, output_data)  # count deviation
    threshold = max(deviations) * 1.1
    neural_network.save(Paths.MODEL_PATH.value.format(data_dir, ml_type, nodes, test_id))
    file = open(Paths.THRESHOLD_PATH.value.format(data_dir, ml_type, nodes, test_id), "w")
    file.write(str(threshold))
    file.close()
    logger.debug("Ending to retrain model for test_id: data_dir: {}, ml_type: {}, nodes: {}, "
                 "test_id: {}.".format(data_dir, ml_type, nodes, test_id))
    return True


def retrain_models_for_nodes(data_dir: str, ml_type: str, nodes: int, test_list: List[str]):
    """
    This function take parameters to identify for which type of data, and number of nodes should
    retrain
    ML model. Then retrains model for all test ids from test_list.
    :param data_dir: Path to root data dir.
    :param ml_type: Type of components(cpus, memory)
    :param nodes: Number of nodes
    :param test_list: List with test ids.
    :return: True if everything went smoothly, in other cases False or throw Exception.
    """
    logger.debug(
        "Starting to retrain models for node: data_dir: {}, ml_type: {}, nodes: {}.".format(
            data_dir, ml_type, nodes))
    if not test_list:
        try:
            test_list = os.listdir(Paths.NODE_DIR.value.format(data_dir, ml_type, nodes))
        except FileNotFoundError as e:
            logger.error("Problem occured to get files from directory {}. Which are needed, "
                         "because no nodes specified. Directory probably doesnt exist or wrong"
                         "permissions".format(Paths.ML_TYPE_DIR.value.format(data_dir, ml_type,
                                                                             nodes)))
            return False
        if not test_list:
            return True
    for test_id in test_list:
        try:
            retrain_model(data_dir, ml_type, nodes, test_id)
        except Exception as e:
            logger.error("Failed retraining model for root data dir: {}, ml_type: {}, nodes: {},"
                         "test_id: {} with error {}.".format(data_dir, ml_type, nodes, test_id, e))
            return False
    logger.debug(
        "Ending to retrain models for node: data_dir: {}, ml_type: {}, nodes: {}.".format(
            data_dir, ml_type, nodes))
    return True


def retrain_models_for_ml_type(data_dir: str, ml_type: str, nodes: List[int], test_list: List[str]):
    """
    This function take parameters to identify type of data and then retrains models for all
    nodes from list of nodes(nodes) and for all test ids from parameter test_list.
    :param data_dir: Path to root data dir.
    :param ml_type: Type of components(cpus, memory)
    :param nodes: List with nodes
    :param test_list: List with test ids.
    :return: True if everything went smoothly, in other cases False or throw Exception.
    """
    logger.debug(
        "Starting to retrain models for ml_type: data_dir: {}, ml_type: {}.".format(data_dir,
                                                                                    ml_type))
    if not nodes:
        try:
            list_with_files = os.listdir(Paths.ML_TYPE_DIR.value.format(data_dir, ml_type))
        except FileNotFoundError:
            logger.error("Problem occured to get files from directory {}. Which are needed, "
                         "because no nodes specified. Directory probably doesnt exist or wrong"
                         "permissions".format(Paths.ML_TYPE_DIR.value.format(data_dir, ml_type)))
            return False

        nodes = list(map(lambda x: int(re.search(r'\d+', x).group(0)), list_with_files))
        if not nodes:
            return True
    for node in nodes:
        retrain_models_for_nodes(data_dir, ml_type, node, test_list)
    logger.debug("Ending to retrain models for ml_type: data_dir: {}, ml_type: {}".format(data_dir,
                                                                                          ml_type))
    return True