from enum import Enum

class MLAlgoType(Enum):
    '''
    Enum to define ML algorithms which are supported by strategy
    '''
    ADABOOST = 1
    RANDOMFOREST = 2
