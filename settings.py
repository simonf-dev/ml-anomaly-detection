from enum import Enum
import logging

from tensorflow import optimizers


class Result(Enum):
    CORRECT = 1
    INCORRECT = 0


class ConstantValues(Enum):
    BOGOMIPS = 4000
    MEM_VALUE = 4000


class InputDataLabels(Enum):
    TIME = "Time"
    IDLE = "cpu_idle_alert"
    STEAL = "cpu_steal_alert"
    SYSTEM = "cpu_sys_alert"
    USER = "cpu_user_alert"
    KSMD = "ksmd_alert"
    MEM_AVA = "mem_ava_alert"


class PlotNames(Enum):
    CPUS = "cpus"
    MEM_AVAILABLE = "mem_available"


class DataFormat(Enum):
    ROW_NUMBER = 1000


class Order(Enum):
    ORDER = {PlotNames.CPUS.value: [InputDataLabels.IDLE.value, InputDataLabels.SYSTEM.value,
                                    InputDataLabels.USER.value, InputDataLabels.KSMD.value, ],
             PlotNames.MEM_AVAILABLE.value:
                 [InputDataLabels.MEM_AVA.value]}


class Paths(Enum):
    MODEL_PATH = "{}/{}/{}node/{}/model"
    THRESHOLD_PATH = "{}/{}/{}node/{}/threshold"
    DATA_DIR = "{}/{}/{}node/{}"
    DATA_FILE = "{}/{}/{}node/{}/{}"
    NODE_DIR = "{}/{}/{}node"
    ML_TYPE_DIR = "{}/{}"
    GNUPLOT_SCRIPT_PATH = "./gnuplot_plotting_script{}.plt"


class LoggingSettings(Enum):
    OUTPUT_FILE = "tmp.log"
    LOGGING_LEVEL = logging.DEBUG


class MLStructureSettings(Enum):
    CPUS = {"lr": 0.0001, "loss": "mse", "optimizer": optimizers.Adam,
            "denses": [{"count": 80, "activation": "relu", "return_sequences": True},
                       {"count": 40, "activation": "relu", "return_sequences": False},
                       {"count": 40, "activation": "relu", "return_sequences": True},
                       {"count": 80, "activation": "relu", "return_sequences": True}]}
    MEMORY = {"lr": 0.0001, "loss": "mse", "optimizer": optimizers.Adam,
              "denses": [{"count": 80, "activation": "relu", "return_sequences": True},
                         {"count": 40, "activation": "relu", "return_sequences": False},
                         {"count": 40, "activation": "relu", "return_sequences": True},
                         {"count": 80, "activation": "relu", "return_sequences": True}]}
