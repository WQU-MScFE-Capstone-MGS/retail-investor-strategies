import operator
from datetime import datetime

from fundamental_data import FundamentalData
from technical_data import TechnicalData
from algo_type import MLAlgoType
from run_config import RunConfig

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


class SmartBetaStrategies(QCAlgorithm):
    '''
    This algo implements Value and Quality stock smart beta strategies.
    User need to set 
    self.value_smart_beta = True for Value strategy
    self.value_smart_beta = False for Quality strategy.
    '''
    def __init__(self):
        self.lookback = 252 # 12 Business or trading Months of data in  business days
        self._lastMonth = 0
        self.fundamentals = {}
        self.technicals = {}
        self.model = {}
        self.columns = []
        self.value_smart_beta = False # make it False to run Quality Smart Beta Strategy
        
    def Initialize(self):
        '''
        This method is responsible to setup the environment and different parameter for algo strategy.
        '''
        # Set Start date for backtesting
        self.SetStartDate(RunConfig.StartDate)
        
        # Set EndDate for backtesting
        self.SetEndDate(RunConfig.EndDate)
        
        self.SetCash(RunConfig.StrategyCash)  # Set Strategy Cash
        
        # Set trade data frequency
        self.UniverseSettings.Resolution = RunConfig.Resolution
        
        # Define benchmark for strategy
        self.bench_index = self.AddEquity(RunConfig.BenchmarkIndex, RunConfig.Resolution).Symbol
        self.SetBenchmark(self.bench_index)
        
        # Initialize the custom collection for strategy
        self.fundametal_data = FundamentalData()
        
        # Set the stock universe for our strategy
        self.AddUniverse(self.Universe.Index.QC500)
        self.SetUniverseSelection(
            FineFundamentalUniverseSelectionModel(self.CoarseSelectionFunction, self.FineSelectionFunction, None, None))
        
        # Set Columns to train ML algorithm
        self.columns = ['PE', 'PB', 'PCF', 'DY', 'CFO', 'ROA', 'ROE', 'DE', 'EPSVar', 'MCAP', 'SP']
        
        # Set Portfolio construction strategy
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        
        # Set Risk management for portfolio
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity())
        
        # Set Execution model for algorithm
        self.SetExecution(ImmediateExecutionModel())
       
        # Note: self.RebalanceOnML If running ML based strategy else self.Rebalance
        # Schedule the algorithm to train and take position 
        if RunConfig.UseMLForRebalancing:
            self.Train(self.DateRules.EveryDay(RunConfig.BenchmarkIndex), self.TimeRules.At(8,0), self.RebalanceOnML)
        else:
            self.Schedule.On(self.DateRules.MonthStart(self.bench_index), self.TimeRules.At(8,0), self.Rebalance)
        
    
    def CoarseSelectionFunction(self, coarse):
        '''
        CoarseSelectionFunction is helper method to feed technical data for selected universe.
        This method filters the stocks based on selected technical criteriaon.
        '''
        
        # Filter stocks which has Fundamental data and price > 5
        CoarseWithFundamental = [x for x in coarse if x.HasFundamentalData and (float(x.Price) > 5)]
    
        # Sort filtered stocks based on dollar volume
        sortedByDollarVolume = sorted(CoarseWithFundamental, key=lambda x: x.DollarVolume, reverse=True)[:500]
        
        # Extract technical data required for strategy and maintain into user defined class
        for cf in sortedByDollarVolume:
            if cf.Symbol.Value not in self.technicals.keys():
                self.technicals[cf.Symbol.Value] = TechnicalData(cf.Symbol.Value)

            # Updates the TechnicalData object with current EOD price
            avg = self.technicals[cf.Symbol.Value]
            avg.update(cf.EndTime, cf.AdjustedPrice)

        return [x.Symbol for x in sortedByDollarVolume]
    
    def FineSelectionFunction(self, fine):
        '''
        FineSelectionFunction is responsible to feed fundamental data to algorithm. 
        Data feed frequency based on resolution set by user.
        '''
        
        # Sort the stocks by market cap.
        top = sorted(fine, key = lambda x: x.EarningReports.BasicAverageShares.ThreeMonths * 
        (x.EarningReports.BasicEPS.TwelveMonths*x.ValuationRatios.PERatio), reverse=True)
        
        # Extract fundamentals for stocks and maintain into user defined class.
        for x in top:
            if x.Symbol.Value not in self.fundametal_data.fundamentals.keys():
                self.fundametal_data.add_symbol(x.Symbol.Value)
            marketcap = abs(x.EarningReports.BasicAverageShares.ThreeMonths * 
                       x.EarningReports.BasicEPS.TwelveMonths * x.ValuationRatios.PERatio)
            
            self.fundametal_data.update_data(x.Symbol.Value,x.EndTime, x.ValuationRatios.PERatio, 
            x.ValuationRatios.PBRatio, x.ValuationRatios.PCFRatio, x.ValuationRatios.TrailingDividendYield,
            x.ValuationRatios.CFOPerShare,x.OperationRatios.ROA.Value, x.OperationRatios.ROE.Value,
            x.OperationRatios.TotalDebtEquityRatio.Value, x.EarningReports.BasicEPS.ThreeMonths, marketcap, 
            x.ValuationRatios.SalesYield, self.technicals[x.Symbol.Value].signal)
            
        return [x.Symbol for x in top]

    def Rebalance(self):
        '''
        Rebalance is empirical rank based trading strategy implementation. It is responsible to take 
        position in portfolio for specific stock.
        This strategy is common implementation for Value Smart Beta and Quality Smart Beta.
        User need to specify 'self.value_smart_beta' before running algorithm.
        Implementation of this strategy is based on Z-Score which defines Value or Quality of stock.
        Top 50 Z-Score stocks are interpreted as long signal. Stocks which are already in portfolio and not part of 
        top 50 will be liquidated.
        '''
        # Initialize collection to store Z-Score
        zScores = {}
        
        # Calculate Z-Score for top 500 stocks.
        for stock in self.fundametal_data.fundamentals.keys():
            if not self.Securities.ContainsKey(stock):
                self.AddEquity(stock)
           
            # Check if stock is tradable and lookback is valid.
            if self.fundametal_data.fundamentals[stock].shape[0] > self.lookback and self.Securities[stock].IsTradable:
                # Calculate Z-Score for individual stock
                z_score = self.fundametal_data.getValueZScore(stock) \
                if self.value_smart_beta else self.fundametal_data.getQualityZScore(stock)
                
                # Store Z-Score into dictionary
                zScores[stock] = z_score
        
        if zScores:
            # Sort Z-Scores in descending order
            sorted_zScore = dict(sorted(zScores.items(), key=operator.itemgetter(1), reverse=True))
            if sorted_zScore:
                # Take Top 50 stocks
                long_stocks = {k: sorted_zScore[k] for k in list(sorted_zScore)[:50]}
                
                # Extract stocks which are in portfolio and invested
                stocks_invested = [x.Key for x in self.Portfolio if x.Value.Invested]
                
                # Liquidate the stocks which are in portfolio but in top 50
                for i in stocks_invested:
                    if i not in long_stocks.keys():
                        self.Liquidate(i)
                
                # Calculate winsor score for all long signaled stocks
                winsor_score = sum([long_stocks[k] for k in long_stocks])
                
                # Go long on top 50 stocks
                for i in long_stocks.keys():
                    if not self.Securities.ContainsKey(i):
                        self.AddEquity(i)
                    self.SetHoldings(i, long_stocks[i]/winsor_score)
    
    def TrainTradingModel(self):
        '''
        TrainTradingModel method will be called on schedule ticks. This method is responsible 
        to fit the data for given ML algo. Method is responsible to fit the model for given stock data.
        User need to ensure which classifier they want to train with.
        '''
        # Train the ML models for all given stocks.
        for k, v in self.fundametal_data.fundamentals.items():
            # Check for lookback and to ensure sufficient data is there for given model.
            # User need to configure lookback before running algo.
            if v.shape[0] < self.lookback:
                continue
            
            # Ensure security is part of list. This is QC framework requirement.
            if not self.Securities.ContainsKey(k):
                self.AddEquity(k)
            
            # Extract data to train the model. X is independent variable array and y is dependent variable
            X = v[self.columns].values
            y = v['Signal'].values
            y=y.astype('float')
            
            # Get the classifier object
            clf = self.Classifier(MLAlgoType.ADABOOST)
            
            # Fit/Train the model for given data
            self.model[k] = clf.fit(X,y)
            
        
    def Classifier(self,ctype):
        '''
        Classifier is Factory method which create the classifier object for given input.
        As of now, there is no optimization parameters provided while creating object for ML algo.
        '''

        # Create AdaBoost classifier object   
        if ctype is MLAlgoType.ADABOOST:
            clf = AdaBoostClassifier()
        
        # Create RandomForest object
        elif ctype is MLAlgoType.RANDOMFOREST:
            clf = RandomForestClassifier(n_estimators=100,max_depth=2)

        else:
            print('no such classifier')
        return clf
    
    def RebalanceOnML(self):
        '''
        RebalanceOnML method is repsonsible to execute trading strategy for given ML algorithm.
        Rank based algorithm is implemented in this strategy. Top 50 stocks which has long signal
        from ML algo are selected. Our strategy will go short for those which are not in top 50 and 
        already in portfolio.
        User need to specify which ML classifier they want to use while calling classifier method.
        We have provided below 3 ML clssifiers to generate the signals.
        1. LogisticRegression
        2. AdaBoost
        3. Random Forest
        
        Instead of implementing these algorithms, we leveraged sklearn library which de facto for ML in python.
        '''
        
        # Initialise variable to store long stocks
        long_stocks = []
        
        # Train the model on scheduled date and time.
        self.TrainTradingModel()
        
        # Identify stocks which our ML algo predicted to go long.
        for k, v in self.fundametal_data.fundamentals.items():
            if not self.Securities.ContainsKey(k):
                self.AddEquity(k)
            if self.Securities[k].IsTradable and k in self.model.keys():
                y = self.model[k].predict(v.tail(1)[self.columns].values)
                if y ==1:
                    long_stocks.append(k)
        
        # Take top 50 stock from long signal stocks
        long_stocks = long_stocks[:50]
        
        # Extract stock which are already in our portfolio 
        stocks_invested = [x.Key for x in self.Portfolio if x.Value.Invested]
        
        # Liquidate stock which are in portfolio but not in top 50
        for i in stocks_invested:
            if i not in long_stocks:
                self.Liquidate(i)
        
        # Go long on new top 50 stocks using equal weighting scheme
        for i in long_stocks:
            if not self.Securities.ContainsKey(i):
                self.AddEquity(i)
            self.SetHoldings(i, 1/len(long_stocks))