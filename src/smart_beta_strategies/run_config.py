from datetime import date

class RunConfig:
    # Start date for backtesting
    StartDate = date(2017, 1, 1)
    
    # End date for backtesting
    EndDate = date(2019, 12, 31)
    
    # Initial Cash
    StrategyCash = 200000
    
    # Resolution of the price data
    Resolution = Resolution.Daily
    
    # Benchmark Index used for evaluation
    BenchmarkIndex = 'SPY'
    
    # Use ML Based rebalancing
    UseMLForRebalancing = False

