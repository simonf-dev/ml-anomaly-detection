from enum import Enum


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
    order = {PlotNames.CPUS.value: [InputDataLabels.IDLE.value, InputDataLabels.SYSTEM.value,
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

