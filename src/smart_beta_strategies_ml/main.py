import operator
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

from fundamental_data import FundamentalData
from algo_type import MLAlgoType
from run_config import RunConfig

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


class SmartBetaStrategies(QCAlgorithm):
    """
    This algo implements Value and Quality stock smart beta strategies.
    User need to set
    self.value_smart_beta = True for Value strategy
    self.value_smart_beta = False for Quality strategy.
    """

    def __init__(self):
        self.lookback = 57  # quarterly rebalance

        self.model = {}  # dict to store ml models

        self.lastMonth = -1  # workaround for monthly universe selection
        self.months = 2  # workaround for quarterly portfolio rebalancing
        self.top = 20  # the amount of stocks to invest

        # Signal parameters
        self.boundary = 1 / 100  # 1%
        self.periods = 20

        self.ml_type = RunConfig.MLType

        self.symbols = []  # list of stocks in the universe to store data
        self.symbols_fine = []  # list of stocks in the universe for portfolio creation

        self.algo_type = "ML"

        self.columns = ['Time', 'PE', 'PB', 'PCF', 'DY', 'CFO', 'ROA', 'ROE', 'DE', 'EPS', 'MCAP', 'SP',
                        'NIGrowth', 'Signal']  # names to store fundamental data
        # Set Columns to train ML algorithm
        self.columns_ml = ['PE', 'PB', 'PCF', 'DY', 'CFO', 'ROA', 'ROE', 'DE', 'EPS', 'MCAP', 'SP', 'NIGrowth']

    def Initialize(self):
        """
        This method is responsible to setup the environment and different parameter for algo strategy.
        """
        # Set Start date for backtesting
        self.SetStartDate(RunConfig.StartDate)  # Note: initial model will be trained with 1 year data

        # Set EndDate for backtesting
        self.SetEndDate(RunConfig.EndDate)

        # Set Strategy Cash
        self.SetCash(RunConfig.StrategyCash)

        # Set trade data frequency
        self.UniverseSettings.Resolution = RunConfig.Resolution

        # Define benchmark for strategy
        self.AddEquity(RunConfig.BenchmarkIndex, RunConfig.Resolution)
        self.SetBenchmark(RunConfig.BenchmarkIndex)

        # Initialize the custom collection for strategy
        self.fundametal_data = FundamentalData(self.columns)

        # Set the stock universe for our strategy
        self.SetUniverseSelection(
            FineFundamentalUniverseSelectionModel(self.CoarseSelectionFunction, self.FineSelectionFunction, None, None))

        # Set Brokerage Model
        self.SetBrokerageModel(InteractiveBrokersBrokerageModel())

        # Set Execution model for algorithm
        self.SetExecution(ImmediateExecutionModel())

        # Schedule the algorithm to train and take position
        self.Schedule.On(self.DateRules.MonthStart(RunConfig.BenchmarkIndex), self.TimeRules.At(8,0), self.RebalanceOnML)

    def CoarseSelectionFunction(self, coarse):
        """
        CoarseSelectionFunction is helper method to feed technical data for selected universe.
        This method filters the stocks based on selected technical criteriaon.
        """

        # workaround for quarterly universe selection
        if self.Time.month == self.lastMonth:
            return self.symbols
        elif self.Time.month % 3 != 0:
            self.lastMonth = self.Time.month
            return self.symbols
        self.lastMonth = self.Time.month

        # Filter stocks which has Fundamental data and price > 5
        CoarseWithFundamental = [x for x in coarse if x.HasFundamentalData and (float(x.Price) > 5)]

        # Sort filtered stocks based on dollar volume
        sortedByDollarVolume = sorted(CoarseWithFundamental, key=lambda x: x.DollarVolume, reverse=True)[:200]

        # save universe for reuse
        self.symbols = [x.Symbol for x in sortedByDollarVolume]

        return self.symbols

    def FineSelectionFunction(self, fine):
        """
        FineSelectionFunction is responsible to feed fundamental data to algorithm.
        Data feed frequency based on resolution set by user.
        """

        # Sort the stocks by market cap.
        top = sorted(fine, key=lambda x: x.EarningReports.BasicAverageShares.ThreeMonths *
                                         (x.EarningReports.BasicEPS.TwelveMonths * x.ValuationRatios.PERatio),
                     reverse=True)

        # Extract fundamentals for stocks and maintain into user defined class.
        for x in top:
            if x.Symbol.Value not in self.fundametal_data.fundamentals.keys():
                self.fundametal_data.add_symbol(x.Symbol.Value)
            marketcap = (x.EarningReports.BasicAverageShares.ThreeMonths *
                         x.EarningReports.BasicEPS.TwelveMonths * x.ValuationRatios.PERatio)
            data = [x.EndTime,
                    x.ValuationRatios.PERatio,
                    x.ValuationRatios.PBRatio, x.ValuationRatios.PCFRatio,
                    x.ValuationRatios.TrailingDividendYield,
                    x.ValuationRatios.CFOPerShare, x.OperationRatios.ROA.Value,
                    x.OperationRatios.ROE.Value,
                    x.OperationRatios.TotalDebtEquityRatio.Value,
                    x.EarningReports.BasicEPS.Value,
                    marketcap,
                    x.ValuationRatios.SalesYield, x.OperationRatios.NetIncomeGrowth.Value,
                    x.Price, ]
            self.fundametal_data.update_data(x.Symbol.Value, data)

        # save symbols for ML
        self.symbols_fine = [x.Symbol.Value for x in top][:100]

        return [x.Symbol for x in top]

    def need_rebalance(self):
        self.months += 1
        if self.months % 3 == 0:  # and self.months >= 12:
            return True
        else:
            return False

    def Rebalance(self):
        """
        Rebalance is empirical rank based trading strategy implementation. It is responsible to take
        position in portfolio for specific stock.
        This strategy is common implementation for Value Smart Beta and Quality Smart Beta.
        User need to specify 'self.value_smart_beta' before running algorithm.
        Implementation of this strategy is based on Z-Score which defines Value or Quality of stock.
        Top 20 Z-Score stocks are interpreted as long signal. Stocks which are already in portfolio and not part of
        top 20 will be liquidated.
        """
        # Initialize collection to store Z-Score
        zScores = {}

        # set initial stocks from position
        if self.set_initial_portfolio:
            self.set_initial_stocks()
            self.set_initial_portfolio = False

        if not self.need_rebalance():
            return

        # Calculate Z-Score for top 100 stocks.
        for stock in self.fundametal_data.fundamentals.keys():
            if not self.Securities.ContainsKey(stock):
                self.AddEquity(stock)

            # Check if stock is tradable and lookback is valid.
            if self.fundametal_data.fundamentals[stock].shape[0] >= self.lookback and self.Securities[stock].IsTradable:
                # Calculate Z-Score for individual stock
                z_score = self.fundametal_data.getValueZScore(stock) \
                    if self.value_smart_beta else self.fundametal_data.getQualityZScore(stock)

                # Store Z-Score into dictionary
                zScores[stock] = z_score

        if zScores:
            # Sort Z-Scores in descending order
            sorted_zScore = dict(sorted(zScores.items(), key=operator.itemgetter(1), reverse=True))
            if sorted_zScore:
                # Take Top 20 stocks
                long_stocks = {k: sorted_zScore[k] for k in list(sorted_zScore)[:self.top]}

                # Extract stocks which are in portfolio and invested
                stocks_invested = [x.Key for x in self.Portfolio if x.Value.Invested]

                # Liquidate the stocks which are in portfolio but in top 20
                for i in stocks_invested:
                    if i.Value not in long_stocks.keys():
                        self.Liquidate(i)

                # Calculate winsor score for all long signaled stocks
                winsor_score = sum([long_stocks[k] for k in long_stocks])

                # Go long on top 20 stocks
                for i in long_stocks.keys():
                    if not self.Securities.ContainsKey(i):
                        self.AddEquity(i)
                    if i in stocks_invested:
                        continue
                    else:
                        self.SetHoldings(i, 0.05)
        self.fundametal_data.fundamentals.clear()



    def TrainTradingModel(self):
        """
        TrainTradingModel method will be called on schedule ticks. This method is responsible
        to fit the data for given ML algo. Method is responsible to fit the model for given stock data.
        User need to ensure which classifier they want to train with.
        """

        # Train the ML models for selected stocks.
        for k in self.symbols_fine:
            d = self.fundametal_data.fundamentals[k]
            # Check for lookback and to ensure sufficient data is there for given model.
            # User need to configure lookback before running algo.
            if len(d['Time']) < self.lookback:
                continue

            # convert dictionary date to dataframe
            v = pd.DataFrame(d)
            v.columns = self.columns
            v.index = v['Time']
            v.fillna(method="ffill")

            # Extract data to train the model. X is independent variable array and y is dependent variable
            X = v[self.columns_ml].values[:-self.periods]

            # signal is 1/0 if return for the next 20 periods exceeds 1% (this parameter can be adjusted)
            signal = (v['Signal'].shift(-self.periods) / v['Signal'] - 1)\
                .apply(lambda x: 1. if x > self.boundary else -1.)
            y = signal.values[:-self.periods]

            # Get the classifier object
            if k in self.model.keys():
                clf = self.model[k]
                if self.ml_type == MLAlgoType.RANDOMFOREST:
                    clf.n_estimators += 2
                # self.Debug(f"{k} has model")
            else:
                clf = self.Classifier(self.ml_type)

            # Fit/Train the model for given data, last value is excluded to remove data leakage
            self.model[k] = clf.fit(X, y)
            if self.ml_type == MLAlgoType.RANDOMFOREST:
                v = v.iloc[-self.periods:]
                self.fundametal_data.fundamentals[k] = v.to_dict(orient='list')

    def Classifier(self, ctype):
        """
        Classifier is Factory method which create the classifier object for given input.
        As of now, there is no optimization parameters provided while creating object for ML algo.
        """

        # Create AdaBoost classifier object   
        if ctype is MLAlgoType.ADABOOST:
            clf = AdaBoostClassifier()

        # Create RandomForest object
        elif ctype is MLAlgoType.RANDOMFOREST:
            clf = RandomForestClassifier(warm_start=True, criterion='entropy',
                                         n_estimators=100, max_depth=2)
        else:
            print('no such classifier')
        return clf

    def Select_by_atr(self, long_stocks):
        """
        Module to select the least volatile stocks by ATR
        """
        res = pd.Series(index=long_stocks)
        history = self.History(20)
        for asset in res.index:
            atr = self.ATR(asset, 14, MovingAverageType.Simple, Resolution.Daily)

            for s in history:
                if s.Bars.ContainsKey(asset):
                    atr.Update(s.Bars[asset])
            if atr.IsReady:
                res[asset] = atr.Current.Value
            else:
                self.Log(f"{asset} hasn't ATR")

        res.dropna(inplace=True)
        res.sort_values(inplace=True)
        return res.index.tolist()[:self.top]

    def RebalanceOnML(self):
        """
        RebalanceOnML method is repsonsible to execute trading strategy for given ML algorithm.
        Rank based algorithm is implemented in this strategy. Top 20 stocks which has long signal
        from ML algo are selected. Our strategy will go short for those which are not in top 20 and
        already in portfolio.
        User need to specify which ML classifier they want to use while calling classifier method.
        We have provided below 2 ML clssifiers to generate the signals.
        1. AdaBoost
        2. Random Forest

        Instead of implementing these algorithms, we leveraged sklearn library which de facto for ML in python.
        """

        if not self.need_rebalance():
            return

        # Initialise variable to store long stocks
        long_stocks = []

        # Train the model on scheduled date and time.
        self.TrainTradingModel()

        # Identify stocks which our ML algo predicted to go long.
        for k in self.symbols_fine:
            v = self.fundametal_data.fundamentals[k]

            # Ensure security is part of list. This is QC framework requirement.
            if not self.Securities.ContainsKey(k):
                self.AddEquity(k)

            if self.Securities[k].IsTradable and k in self.model.keys():

                try:
                    # get the last value of fundamentals features
                    X = [v[i][-1] for i in self.columns_ml]
                    X = np.array(X).reshape(1, -1)
                    y = self.model[k].predict(X)

                    if y == 1:
                        long_stocks.append(k)
                except ValueError as e:
                    self.Log(f"Error {k, len(v['Time']), e}")

        # Select least volatile 20 stocks from long signal stocks
        long_stocks = self.Select_by_atr(long_stocks)
        self.Log(f"long {self.top} stocks: {long_stocks}")

        # Extract stock which are already in our portfolio 
        stocks_invested = [x.Key for x in self.Portfolio if x.Value.Invested]

        # Liquidate stock which are in portfolio but not in top 50
        for i in stocks_invested:
            if i.Value not in long_stocks:
                self.Liquidate(i)

        # Go long on new top 20 stocks using equal weighting scheme
        for i in long_stocks:
            if not self.Securities.ContainsKey(i):
                self.AddEquity(i)
            self.SetHoldings(i, 1 / len(long_stocks))
