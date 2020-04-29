import pandas as pd
from collections import deque


class FundamentalData(object):
    '''
    FundamentalData class is responsible to handle fundamental data of stocks.
    This class collects and maintain the fundamental data and perforam different operations on same.
    '''

    def __init__(self, columns, algo="ML"):
        self.columns = columns
        """['Time', 'PE', 'PB', 'PCF', 'DY', 'CFO', 'ROA', 'ROE', 'DE', 'EPS', 'MCAP', 'SP',
                        'NIGrowth', 'Signal']"""
        self.algo = algo
        self.fundamentals = {}

    def add_symbol(self, symbol):
        self.fundamentals[symbol] = {i: deque(maxlen=1000) for i in self.columns}

    def update_data(self, symbol, data):
        for name, value in zip(self.columns, data):
            self.fundamentals[symbol][name].append(value)