import pandas as pd
from clr import AddReference

AddReference("System")
AddReference("QuantConnect.Algorithm")
AddReference("QuantConnect.Common")
AddReference("QuantConnect.Indicators")

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Algorithm.Framework import *
from QuantConnect.Algorithm.Framework.Risk import *
from QuantConnect.Algorithm.Framework.Alphas import *
from QuantConnect.Algorithm.Framework.Execution import *
from QuantConnect.Algorithm.Framework.Portfolio import *
from QuantConnect.Algorithm.Framework.Selection import *

from QuantConnect.Data import SubscriptionDataSource
from QuantConnect.Python import PythonData

from datetime import timedelta, datetime, date
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample, shuffle


class RunConfig:
    """This class contains major parameters for running algorithm
    """

    # Start date for back-testing
    StartDate = date(2009, 5, 1)

    # End date for backtesting
    EndDate = date(2019, 12, 1)

    # Initial Cash
    StrategyCash = 200000

    # Selection of long only (True) or long-short (False)
    LongOnly = True

    # Position holding period, days
    PositionLifetime = timedelta(days=25)

    # Vertical barrier, days (25 or 35 days for QC platform)
    VertBarDays = 25

    # For running on LEAN locally please provide a path to folder with data
    PathToLocalFolder = ""


class QCTickDataStrategy(QCAlgorithm):
    """ This algo implements RF triple barrier strategy based on raw tick data.
    """

    def __init__(self):

        # symbols of assets from MOEX
        self.assets_keys = ['AFKS', 'ALRS', 'CHMF', 'GAZP',
                            'GMKN', 'LKOH', 'MGNT', 'MTSS',
                            'NVTK', 'ROSN', 'RTKM', 'SBER',
                            'SNGS', 'TATN', 'VTBR', 'YNDX']

        # features to store in dataframe for ML
        self.colsU = ['Logret', 'Momone', 'Momtwo', 'Momthree', 'Momfour', 'Momfive',
                      'Volatilityfifty', 'Volatilitythirtyone', 'Volatilityfifteen',
                      'Autocorrone', 'Autocorrtwo', 'Autocorrthree', 'Autocorrfour', 'Autocorrfive',
                      'Logtone', 'Logttwo', 'Logtthree', 'Logtfour', 'Logtfive',
                      'Bin', 'Side']

        # dictionary to store custom asset objects
        self.assets = {}

        # dictionary to store pandas DataFrames with features for ML
        self.features_dict = {}

        # dictionary to store ML classifier (RandomForest)
        self.clf_dict = {}

        # dictionary to store end holding time for each position
        self.stop_time_dict = {}

    def Initialize(self):
        # setting start and end date to run algorithm
        self.SetStartDate(RunConfig.StartDate)
        self.SetEndDate(RunConfig.EndDate)

        # setting initial funds
        self.SetCash(RunConfig.StrategyCash)

        # creating custom assets from AdvancedBars class for each symbol
        self.assets = {i: self.AddData(AdvancedBars, i) for i in self.assets_keys}

        # creating empty dataframes for each symbol
        self.features_dict = {i: pd.DataFrame(columns=self.colsU) for i in self.assets_keys}

        # creating a dictionary of classifiers with initial None value
        self.clf_dict = {i: None for i in self.assets_keys}

        # creating a dictionary with stoptimes for each symbol
        self.stop_time_dict = {i: self.Time for i in self.assets_keys}

        # setting a schedule to run ML training
        self.Schedule.On(self.DateRules.MonthStart(), self.TimeRules.At(13, 10), Action(self.TrainML))

    def OnData(self, data):
        """OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        """

        for k in self.assets_keys:
            cond1 = (self.Time > self.stop_time_dict[k])  # if holding time is over
            cond2 = self.Portfolio[k].Invested  # if the position open
            if cond1 and cond2:
                self.Log(f", {k}, to liq, {self.Portfolio[k].Quantity}, {self.Portfolio[k].Price}")
                self.Liquidate(k)  # liquidate open position

        for k in self.assets_keys:
            if not data.ContainsKey(k):
                continue

            dat = data[k]
            time = dat.Time

            # saving data into feature for ML
            try:
                self.features_dict[k].loc[time] = [dat.Logret, dat.Momone, dat.Momtwo, dat.Momthree, dat.Momfour,
                                                   dat.Momfive, dat.Volatilityfifty, dat.Volatilitythirtyone,
                                                   dat.Volatilityfifteen,
                                                   dat.Autocorrone, dat.Autocorrtwo, dat.Autocorrthree,
                                                   dat.Autocorrfour, dat.Autocorrfive,
                                                   dat.Logtone, dat.Logttwo, dat.Logtthree, dat.Logtfour, dat.Logtfive,
                                                   dat.Bin, dat.Side]
            except AttributeError as e:
                continue

            if self.clf_dict[k] is not None:  # feed data into ML if RF classifier was created

                # features
                X = self.features_dict[k].drop(["Bin"], axis=1).loc[time].values.reshape(1, -1)

                # predicted value
                y_pred = self.clf_dict[k].predict(X)

                if y_pred > .8:  # for probably enough decision to trade

                    # decision of trade direction is based on sma
                    if dat.Side == 1:  # long position
                        # set new stop time for position holding
                        self.stop_time_dict[k] = self.Time + RunConfig.PositionLifetime

                        if not self.Portfolio[k].IsLong:  # if no long position invested
                            if self.Portfolio[k].Invested:  # if short position
                                self.Liquidate(k)
                        else:
                            continue

                    elif dat.Side == -1 and not RunConfig.LongOnly:  # if not long only portfolio, and if short side
                        # set new stop time for position holding
                        self.stop_time_dict[k] = self.Time + RunConfig.PositionLifetime

                        if self.Portfolio[k].IsLong:
                            self.Liquidate(k)

                    else:
                        continue

                    size = dat.Side * min((self.Portfolio.Cash / self.Portfolio.TotalPortfolioValue) * 0.90, 0.1)
                    self.SetHoldings(k, size)

                    # store trade to log
                    self.Log(f", {k}, pos, {self.Portfolio[k].Quantity}, {self.Portfolio[k].Price}")

    def Balancing(self, X, y):
        """Module to make equial amount of labels. This code is sampled from 'mlfinlab' package
        """

        train_df = pd.concat([y, X], axis=1, join='inner')

        # Upsample the training data to have a 50 - 50 split
        # https://elitedatascience.com/imbalanced-classes
        majority = train_df[train_df['Bin'] == 0]
        minority = train_df[train_df['Bin'] == 1]

        if len(majority) < len(minority):
            majority, minority = minority, majority

        new_minority = resample(minority,
                                replace=True,  # sample with replacement
                                n_samples=majority.shape[0],  # to match majority class
                                random_state=42)
        train_df = pd.concat([majority, new_minority])
        train_df = shuffle(train_df, random_state=42)

        # Create training data
        y_train = train_df['Bin']
        X_train = train_df.loc[:, train_df.columns != 'Bin']

        return X_train, y_train

    def TrainML(self):
        """This module to train RF with new data
        """
        # re-learn ML quarterly
        if self.Time.month % 3 != 0:
            return

        for k in self.assets_keys:
            a = self.features_dict[k].shape
            # self.Debug(f"{self.Time} asset: {k} shape: {a}")

            if a[0] > 800:

                df = self.features_dict[k].copy()

                # ensure no leakage from future
                df = df[df.index < (self.Time - timedelta(days=RunConfig.VertBarDays))]
                X = df.drop(["Bin"], axis=1)
                y = df["Bin"]

                X, y = self.Balancing(X, y)  # make equal amount of each label

                if self.clf_dict[k] is None:  # for initial creation of RF
                    n_estimator, depth = 100, 3
                    self.clf_dict[k] = RandomForestClassifier(max_depth=depth,
                                                              n_estimators=n_estimator,
                                                              criterion='entropy',
                                                              random_state=42,
                                                              n_jobs=-1)

                self.clf_dict[k].fit(X, y)


class AdvancedBars(PythonData):
    """Custom DollarBars from external files
    """

    def GetSource(self, config, date, isLiveMode):
        """Standard method to read data
        """
        if RunConfig.PathToLocalFolder == "":
            if RunConfig.VertBarDays == 35:
                data = dict(
                    AFKS="https://www.dropbox.com/s/xhq8y82wpm3kc2c/AFKS_35_10.0_20-50_ind.csv?dl=1",
                    ALRS="https://www.dropbox.com/s/n2uvuof4o7su6gy/ALRS_35_10.0_20-50_ind.csv?dl=1",
                    CHMF="https://www.dropbox.com/s/pzobkz3euks2j99/CHMF_35_10.0_20-50_ind.csv?dl=1",
                    GAZP="https://www.dropbox.com/s/0j5ljcrot13lyqg/GAZP_35_10.0_20-50_ind.csv?dl=1",
                    GMKN="https://www.dropbox.com/s/9u7vh4qtgtsbqa5/GMKN_35_10.0_20-50_ind.csv?dl=1",
                    LKOH="https://www.dropbox.com/s/79llbhzk488tz6t/LKOH_35_10.0_20-50_ind.csv?dl=1",
                    MGNT="https://www.dropbox.com/s/fs5e9u3rh3wu7wa/MGNT_35_10.0_20-50_ind.csv?dl=1",
                    MTSS="https://www.dropbox.com/s/ipqow24i276r0u8/MTSS_35_10.0_20-50_ind.csv?dl=1",
                    NVTK="https://www.dropbox.com/s/03ypu9xugy2ty09/NVTK_35_10.0_20-50_ind.csv?dl=1",
                    ROSN="https://www.dropbox.com/s/93lrc7vwdshoxj2/ROSN_35_10.0_20-50_ind.csv?dl=1",
                    RTKM="https://www.dropbox.com/s/rlhdm1m50slwo0s/RTKM_35_10.0_20-50_ind.csv?dl=1",
                    SBER="https://www.dropbox.com/s/whi42pzugxp39o3/SBER_35_10.0_20-50_ind.csv?dl=1",
                    SNGS="https://www.dropbox.com/s/akjpwuxxg2gs847/SNGS_35_10.0_20-50_ind.csv?dl=1",
                    TATN="https://www.dropbox.com/s/zzezjtpp5756iuw/TATN_35_10.0_20-50_ind.csv?dl=1",
                    VTBR="https://www.dropbox.com/s/rvg93vazaiqqghu/VTBR_35_10.0_20-50_ind.csv?dl=1",
                    YNDX="https://www.dropbox.com/s/od0m2jrewb222ze/YNDX_35_10.0_20-50_ind.csv?dl=1",
                )
            else:
                data = dict(
                    AFKS="https://www.dropbox.com/s/khemhea62b9uenv/AFKS_25_10.0_20-50_ind.csv?dl=1",
                    ALRS="https://www.dropbox.com/s/ahhl110ojc2z6xt/ALRS_25_10.0_20-50_ind.csv?dl=1",
                    CHMF="https://www.dropbox.com/s/0mrzkhf0pb5vxeg/CHMF_25_10.0_20-50_ind.csv?dl=1",
                    GAZP="https://www.dropbox.com/s/ybowz18784joy3r/GAZP_25_10.0_20-50_ind.csv?dl=1",
                    GMKN="https://www.dropbox.com/s/2cm5cvr5xef6yut/GMKN_25_10.0_20-50_ind.csv?dl=1",
                    LKOH="https://www.dropbox.com/s/ahfcpqqrkliqxi9/LKOH_25_10.0_20-50_ind.csv?dl=1",
                    MGNT="https://www.dropbox.com/s/qzjjtqd4s6glwmh/MGNT_25_10.0_20-50_ind.csv?dl=1",
                    MTSS="https://www.dropbox.com/s/hq7m2ybfmxzqrti/MTSS_25_10.0_20-50_ind.csv?dl=1",
                    NVTK="https://www.dropbox.com/s/l07wcjrob9q83jx/NVTK_25_10.0_20-50_ind.csv?dl=1",
                    ROSN="https://www.dropbox.com/s/ou006lgl7misb20/ROSN_25_10.0_20-50_ind.csv?dl=1",
                    RTKM="https://www.dropbox.com/s/sc929uafwtbnust/RTKM_25_10.0_20-50_ind.csv?dl=1",
                    SBER="https://www.dropbox.com/s/jmf6ncneworo22x/SBER_25_10.0_20-50_ind.csv?dl=1",
                    SNGS="https://www.dropbox.com/s/yjvxoyj1qz2p79n/SNGS_25_10.0_20-50_ind.csv?dl=1",
                    TATN="https://www.dropbox.com/s/mp76ajmme92dyeb/TATN_25_10.0_20-50_ind.csv?dl=1",
                    VTBR="https://www.dropbox.com/s/hpy3zp6gpfi7rru/VTBR_25_10.0_20-50_ind.csv?dl=1",
                    YNDX="https://www.dropbox.com/s/zm3grnbr62t8hqb/YNDX_25_10.0_20-50_ind.csv?dl=1",
                )

            path = data[config.Symbol.Value]

            return SubscriptionDataSource(path, SubscriptionTransportMedium.RemoteFile);

        else:
            path = RunConfig.PathToLocalFolder + \
                   f"/{config.Symbol.Value}_{str(RunConfig.VertBarDays)}_10.0_20-50_ind.csv"

            return SubscriptionDataSource(path, SubscriptionTransportMedium.RemoteFile);

    def Reader(self, config, line, date, isLiveMode):
        """Standard QC data processing method
        """
        bar = AdvancedBars()
        bar.Symbol = config.Symbol

        if not (line.strip() and line[0].isdigit()): return None

        # data in downloaded file
        cols = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Fastmavg', 'Slowmavg',
                'Sideold', 'Logret', 'Momone', 'Momtwo', 'Momthree', 'Momfour', 'Momfive',
                'Volatilityfifty', 'Volatilitythirtyone', 'Volatilityfifteen',
                'Autocorrone', 'Autocorrtwo', 'Autocorrthree', 'Autocorrfour', 'Autocorrfive',
                'Logtone', 'Logttwo', 'Logtthree', 'Logtfour', 'Logtfive',
                'Ret', 'Trgt', 'Bin', 'Side']

        try:
            data = line.split(',')

            bar.Time = datetime.strptime(data[0], "%Y-%m-%d %H:%M:%S.%f")
            bar.Value = float(data[4])

            for j, c in enumerate(cols):
                if (data[j] != '') and (j != 0):
                    try:
                        bar[c] = float(data[j])
                    except (ValueError, IndexError) as e:
                        pass

            return bar;

        except ValueError:
            return None
