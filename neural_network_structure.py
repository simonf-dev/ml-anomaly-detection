import os
from typing import List

from option_parsers import get_logger
from settings import LoggingSettings, DataFormat, Order, PlotNames, MLStructureSettings

logger = get_logger(LoggingSettings.LOGGING_LEVEL.value, LoggingSettings.OUTPUT_FILE.value,
                    "neural_network_structure")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import optimizers
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
import pandas as pd


def get_neural_network(ml_type: str) -> Sequential:
    """
    This functions returns ML structure for concrete type of data.
    :param ml_type: Type of ML, that we want to get. It could be different in number of dimensions
    etc.
    :return: ML structure for concrete type.
    """
    if ml_type == PlotNames.MEM_AVAILABLE.value:
        time_steps = DataFormat.ROW_NUMBER.value
        features = len(Order.ORDER.value[PlotNames.MEM_AVAILABLE.value])
        denses = MLStructureSettings.MEMORY.value["denses"]
        lstm_autoencoder = Sequential()
        lstm_autoencoder.add(LSTM(denses[0]["count"], activation=denses[0]["activation"],
                                  input_shape=(time_steps, features),
                                  return_sequences=denses[0]["return_sequences"]))
        lstm_autoencoder.add(LSTM(denses[1]["count"], activation=denses[1]["activation"],
                                  return_sequences=denses[1]["return_sequences"]))
        lstm_autoencoder.add(RepeatVector(time_steps))
        lstm_autoencoder.add(LSTM(denses[2]["count"],
                                  activation=denses[2]["activation"],
                                  return_sequences=denses[2]["return_sequences"]))
        lstm_autoencoder.add(LSTM(denses[3]["count"], activation=denses[3]["activation"],
                                  return_sequences=denses[3]["return_sequences"]))
        lstm_autoencoder.add(TimeDistributed(Dense(features)))

        lstm_autoencoder.compile(loss=MLStructureSettings.MEMORY.value["loss"],
                                 optimizer=MLStructureSettings.MEMORY.value["optimizer"])
    elif ml_type == PlotNames.CPUS.value:
        time_steps = DataFormat.ROW_NUMBER.value
        features = len(Order.ORDER.value[PlotNames.CPUS.value])
        denses = MLStructureSettings.CPUS.value["denses"]
        lstm_autoencoder = Sequential()
        lstm_autoencoder.add(LSTM(denses[0]["count"], activation=denses[0]["activation"],
                                  input_shape=(time_steps, features),
                                  return_sequences=denses[0]["return_sequences"]))
        lstm_autoencoder.add(LSTM(denses[1]["count"], activation=denses[1]["activation"],
                                  return_sequences=denses[1]["return_sequences"]))
        lstm_autoencoder.add(RepeatVector(time_steps))
        lstm_autoencoder.add(LSTM(denses[2]["count"],
                                  activation=denses[2]["activation"],
                                  return_sequences=denses[2]["return_sequences"]))
        lstm_autoencoder.add(LSTM(denses[3]["count"], activation=denses[3]["activation"],
                                  return_sequences=denses[3]["return_sequences"]))
        lstm_autoencoder.add(TimeDistributed(Dense(features)))

        lstm_autoencoder.compile(loss=MLStructureSettings.CPUS.value["loss"],
                                 optimizer=MLStructureSettings.CPUS.value["optimizer"])
    else:
        logger.error("Undefined type of ML.")
        raise NotImplementedError
    return lstm_autoencoder
