from datetime import date
from algo_type import MLAlgoType

class RunConfig:
    # Start date for backtesting
    StartDate = date(2010, 1, 10)
    # StartDate = date(2008, 1, 1)
    
    # End date for backtesting
    EndDate = date(2020, 1, 9)
    # EndDate = date(2010, 3, 30)
    
    # Initial Cash
    StrategyCash = 100000
    
    # Resolution of the price data
    Resolution = Resolution.Daily
    
    # Benchmark Index used for evaluation
    BenchmarkIndex = 'SPY'
    
    # MLAlgoType: 'MLAlgoType.RANDOMFOREST or MLAlgoType.ADABOOST
    MLType = MLAlgoType.RANDOMFOREST # MLAlgoType.ADABOOST # 
    
    # Run preliminary data gethering and initial model fitting if True
    #PreliminaryFit = False