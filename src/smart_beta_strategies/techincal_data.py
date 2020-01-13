# Your New Python File
class TechnicalData(object):
    '''
    TechnicalData class is responsible to handle technical data for given stock.
    This user defined class is used to calculate SMA to identify the cross to generate the signals
    '''
    def __init__(self, symbol):
        '''
        Constructor to initialize members of class
        '''
        self.symbol = symbol
        self.fast = SimpleMovingAverage(20)
        self.slow = SimpleMovingAverage(60)
        self.signal = 1

    def update(self, time, value):
        '''
        Update the signal on arrival of new data.
        '''
        if self.fast.Update(time, value) and self.slow.Update(time, value):
            fast = self.fast.Current.Value
            slow = self.slow.Current.Value
            self.signal = 1 if fast > slow else -1
