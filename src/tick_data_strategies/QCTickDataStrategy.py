
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
from sklearn.model_selection import GridSearchCV


class QCTickDataStrategy(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2009, 5, 2)
        self.SetEndDate(2010, 12, 13)
        self.SetCash(100000)

        self.assets_keys = ['AFKS', 'ALRS', 'CHMF', 'GAZP',
                            'GMKN', 'LKOH', 'MGNT', 'MTSS',
                            'NVTK', 'ROSN', 'RTKM', 'SBER',
                            'SNGS', 'TATN', 'VRBR', 'YNDX']

        self.assets = {i: self.AddData(AdvancedBars, i) for i in self.assets_keys}
        # self.AddData(AdvancedBars, "GAZP")

        self.Schedule.On(self.DateRules.MonthStart(), self.TimeRules.At(13, 10), Action(self.TrainML))

        colsU = [  # 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Fastmavg', 'Slowmavg', 'Sideold',
            'Logret', 'Momone', 'Momtwo', 'Momthree', 'Momfour', 'Momfive',
            'Volatilityfifty', 'Volatilitythirtyone', 'Volatilityfifteen',
            'Autocorrone', 'Autocorrtwo', 'Autocorrthree', 'Autocorrfour', 'Autocorrfive',
            'Logtone', 'Logttwo', 'Logtthree', 'Logtfour', 'Logtfive',
            # 'Ret', 'Trgt',
            'Bin', 'Side']

        self.features_dict = {i: pd.DataFrame(columns=colsU) for i in self.assets_keys}
        # self.features = pd.DataFrame(columns=colsU)

        self.clf_dict = {i: None for i in self.assets_keys}
        # self.clf = None

        self.changed = False
        self.long = True

        self.lifetime = timedelta(days=10)
        self.stop_time_dict = {i: self.Time for i in self.assets_keys}
        # self.stop_time = self.Time

        # self.AddData(BenchmarkMOEX, "MOEX", Resolution.Daily)
        # self.SetBenchmark(BenchmarkMOEX, "MOEX")

    def OnData(self, data):
        """OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        """

        for k in self.assets_keys:
            cond1 = (self.Time > self.stop_time_dict[k])
            cond2 = self.Portfolio[k].Invested
            # self.Debug(f"cond1 {cond1}, cond2 {cond2}")
            if cond1 and cond2:
                self.Debug(f"{self.Time}, {k} position {self.Portfolio[k].Quantity}")
                self.Liquidate(k)
                self.Debug(f"{k} position liquidated: {self.Portfolio[k].Quantity}")

        for k in self.assets_keys:
            if not data.ContainsKey(k):
                continue

            dat = data[k]
            time = dat.Time

            try:
                # self.features.loc[time] = [data["GAZP"].Fastmavg, data["GAZP"].Slowmavg, data["GAZP"].Close]
                # self.features.loc[time]
                self.features_dict[k].loc[time] = [dat.Logret, dat.Momone, dat.Momtwo, dat.Momthree, dat.Momfour,
                                                   dat.Momfive, dat.Volatilityfifty, dat.Volatilitythirtyone,
                                                   dat.Volatilityfifteen,
                                                   dat.Autocorrone, dat.Autocorrtwo, dat.Autocorrthree,
                                                   dat.Autocorrfour, dat.Autocorrfive,
                                                   dat.Logtone, dat.Logttwo, dat.Logtthree, dat.Logtfour, dat.Logtfive,
                                                   dat.Bin, dat.Side]
                # self.Debug("1")
            except AttributeError as e:
                continue

            if self.clf_dict[k] is not None:
                X = self.features_dict[k].drop(["Bin"], axis=1).loc[time].values.reshape(1, -1)
                y_pred = self.clf_dict[k].predict(X)

                if y_pred > .8:

                    if dat.Side == 1:
                        if not self.Portfolio[k].IsLong:
                            self.stop_time_dict[k] = self.Time + self.lifetime
                            if self.Portfolio[k].Invested:
                                self.Liquidate(k)
                            self.SetHoldings(k, .5)
                            # self.Debug(f" long {k}, {self.Portfolio[k].Quantity}, till {self.stop_time_dict[k]}")
                            # self.Debug(f" hol {self.Portfolio.TotalHoldingsValue}, cash {self.Portfolio.Cash}")

                        else:
                            self.stop_time_dict[k] = self.Time + self.lifetime
                            # self.Debug(f" long_ {k}, {self.Portfolio[k].Quantity}, till {self.stop_time_dict[k]}")

                    elif dat.Side == -1:
                        if self.Portfolio[k].IsLong:
                            self.stop_time_dict[k] = self.Time + self.lifetime
                            self.Liquidate(k)
                            self.SetHoldings(k, -0.5)
                            # self.Debug(f" short {k}, {self.Portfolio[k].Quantity}, till {self.stop_time_dict[k]}")
                            # self.Debug(f" hol {self.Portfolio.TotalHoldingsValue}, cash {self.Portfolio.Cash}")
                        else:
                            self.stop_time_dict[k] = self.Time + self.lifetime
                            # self.Liquidate(k)
                            self.SetHoldings(k, -0.5)
                            # self.Debug(f" short_ {k}, {self.Portfolio[k].Quantity}, till {self.stop_time_dict[k]}")
                            # self.Debug(f" hol {self.Portfolio.TotalHoldingsValue}, cash {self.Portfolio.Cash}")


    def TrainML(self):

        # re-learn ML quarterly
        if self.Time.month % 3 != 0:
            return

        for k in self.assets_keys:
            a = self.features_dict[k].shape

            # self.Debug(f"{self.Time} asset: {k} shape: {a}")

            if a[0] > 800:

                X = self.features_dict[k].drop(["Bin"], axis=1).values
                y = self.features_dict[k]["Bin"].values.ravel()

                if self.clf_dict[k] is None:
                    n_estimator, depth = self._GridSearchML(X, y)
                    self.clf_dict[k] = RandomForestClassifier(max_depth=depth,
                                                              n_estimators=n_estimator,
                                                              criterion='entropy', random_state=42)

                self.clf_dict[k].fit(X, y)
        # self.Debug(f"Training ML data shape {self.features.shape}, before {a}")

    def _GridSearchML(self, X, y):
        parameters = {'max_depth': [2, 3, 4, 5, 7],
                      'n_estimators': [1, 10, 25, 50, 100, 256, 512],
                      'random_state': [42]}

        rf = RandomForestClassifier(criterion='entropy')

        grid_search_cv = GridSearchCV(rf, parameters, cv=4, scoring='roc_auc', n_jobs=3)

        # self.clf = grid_search_cv.best_estimator_

        grid_search_cv.fit(X, y)

        # self.Log(f"Grid search mean test score {grid_search_cv.cv_results_['mean_test_score']}")

        return grid_search_cv.best_params_['n_estimators'], grid_search_cv.best_params_['max_depth']


class AdvancedBars(PythonData):
    """Custom advanced bars (DollarBars, VolumeBars, etc)"""

    def GetSource(self, config, date, isLiveMode):

        data = dict(
            AFKS="https://www.dropbox.com/s/sltdv01h2qagsuz/AFKS_10_0.1_indicators.csv?dl=1",
            ALRS="https://www.dropbox.com/s/kvv41qbqkvae57z/ALRS_10_0.1_indicators.csv?dl=1",
            CHMF="https://www.dropbox.com/s/o3zllu8roi5mul0/CHMF_10_0.1_indicators.csv?dl=1",
            GAZP="https://www.dropbox.com/s/8ok7497uzojp719/GAZP_10_0.1_indicators.csv?dl=1",
            GMKN="https://www.dropbox.com/s/xt3pda9hjgoitu7/GMKN_10_0.1_indicators.csv?dl=1",
            LKOH="https://www.dropbox.com/s/v747cnwefjlwx5a/LKOH_10_0.1_indicators.csv?dl=1",
            MGNT="https://www.dropbox.com/s/ppgtj7jgme0nasb/MGNT_10_0.1_indicators.csv?dl=1",
            MTSS="https://www.dropbox.com/s/gnti48z0ar1isz4/MTSS_10_0.1_indicators.csv?dl=1",
            NVTK="https://www.dropbox.com/s/9veawbt7awy6avv/NVTK_10_0.1_indicators.csv?dl=1",
            ROSN="https://www.dropbox.com/s/hpzh2pm5cluuohn/ROSN_10_0.1_indicators.csv?dl=1",
            RTKM="https://www.dropbox.com/s/9q61fc2rx5psbor/RTKM_10_0.1_indicators.csv?dl=1",
            SBER="https://www.dropbox.com/s/x4qs21ocx8imq0k/SBER_10_0.1_indicators.csv?dl=1",
            SNGS="https://www.dropbox.com/s/114l8afw1t66pxo/SNGS_10_0.1_indicators.csv?dl=1",
            TATN="https://www.dropbox.com/s/y55z0ay09v6qrfh/TATN_10_0.1_indicators.csv?dl=1",
            VRBR="https://www.dropbox.com/s/erao89h62f8aal8/VTBR_10_0.1_indicators.csv?dl=1",
            YNDX="https://www.dropbox.com/s/qhr27lo71m0fk3c/YNDX_10_0.1_indicators.csv?dl=1"
        )

        path = data[config.Symbol.Value]

        """
        path = "...." + config.Symbol.Value + "_10_0.1_indicators.csv"
        """

        return SubscriptionDataSource(path, SubscriptionTransportMedium.RemoteFile);

    def Reader(self, config, line, date, isLiveMode):
        bar = AdvancedBars()
        bar.Symbol = config.Symbol

        if not (line.strip() and line[0].isdigit()): return None

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
            # bar['Length'] = len(data)

            for j, c in enumerate(cols):
                if (data[j] != '') and (j != 0):
                    try:
                        bar[c] = float(data[j])
                    except (ValueError, IndexError) as e:
                        pass

            return bar;

        except ValueError:
            return None


class BenchmarkMOEX(PythonData):
    """Custom benchmark data
    """

    def GetSource(self, config, date, isLiveMode):

        path = "https://www.dropbox.com/s/fl9oe733ls677wv/RI.IMOEX_090101_191213.csv?dl=1"
        return SubscriptionDataSource(path, SubscriptionTransportMedium.RemoteFile);

    def Reader(self, config, line, date, isLiveMode):
        bar = BenchmarkMOEX()
        bar.Symbol = config.Symbol

        if not (line.strip() and line[0] == "R"): return None

        try:
            data = line.split(';')

            bar.Time = datetime.strptime(data[2], "%Y%m%d")
            bar.Value = float(data[7])
            bar.Open = float(data[4])
            bar.High = float(data[5])
            bar.Low = float(data[6])
            bar.Close = float(data[7])

            return bar

        except ValueError:
            return None
