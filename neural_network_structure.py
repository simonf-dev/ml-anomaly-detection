import logging
import os
from typing import List
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import optimizers
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
import pandas as pd

from settings import DataFormat, Order, PlotNames
lr = 0.0001

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
def prepare_data_for_neural_network(dir_path: str, order: List[str]):
    datasets = []
    for value in order:
        dataset = pd.read_csv("{}/{}".format(dir_path,order))


def get_neural_network(ml_type: str):
    if ml_type == PlotNames.MEM_AVAILABLE.value:
        timesteps = DataFormat.ROW_NUMBER.value
        features = len(Order.order.value[PlotNames.MEM_AVAILABLE.value])
        lstm_autoencoder = Sequential()
        lstm_autoencoder.add(LSTM(80, activation='relu', input_shape=(timesteps,
                                                                      features),
                                  return_sequences=True))
        lstm_autoencoder.add(LSTM(40, activation='relu', return_sequences=False))
        lstm_autoencoder.add(RepeatVector(timesteps))
        # Decoder
        lstm_autoencoder.add(LSTM(40, activation='relu', return_sequences=True))
        lstm_autoencoder.add(LSTM(80, activation='relu', return_sequences=True))
        lstm_autoencoder.add(TimeDistributed(Dense(features)))
        adam = optimizers.Adam(lr)

        lstm_autoencoder.compile(loss='mse', optimizer=adam)
    if ml_type == PlotNames.CPUS.value:
        timesteps = DataFormat.ROW_NUMBER.value
        features = len(Order.order.value[PlotNames.CPUS.value])
        lstm_autoencoder = Sequential()
        lstm_autoencoder.add(LSTM(80, activation='relu', input_shape=(timesteps,
                                                                      features),
                                  return_sequences=True))
        lstm_autoencoder.add(LSTM(40, activation='relu', return_sequences=False))
        lstm_autoencoder.add(RepeatVector(timesteps))
        # Decoder
        lstm_autoencoder.add(LSTM(40, activation='relu', return_sequences=True))
        lstm_autoencoder.add(LSTM(80, activation='relu', return_sequences=True))
        lstm_autoencoder.add(TimeDistributed(Dense(features)))
        adam = optimizers.Adam(lr)

        lstm_autoencoder.compile(loss='mse', optimizer=adam)
    return lstm_autoencoder