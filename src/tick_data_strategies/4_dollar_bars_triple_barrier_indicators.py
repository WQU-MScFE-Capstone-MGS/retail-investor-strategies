#!/usr/bin/env python
# coding: utf-8

import time
import sys
import os
import numpy as np
import pandas as pd
import datetime as dt

import multiprocessing as mp

import mlfinlab as ml

np.random.seed(42)

# module to substitute in 'mlfinlab' package
def new_batch_run(self, verbose=True, to_csv=False, output_path=None):
    """
    Reads a csv file in batches and then constructs the financial data structure in the form of a DataFrame.
    The csv file must have only 3 columns: date_time, price, & volume.
    :param verbose: (Boolean) Flag whether to print message on each processed batch or not
    :param to_csv: (Boolean) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
    :param output_path: (Boolean) Path to results file, if to_csv = True

    :return: (DataFrame or None) Financial data structure
    """

    # for parquet
    if ".gzip" in self.file_path:
        parquet = pd.read_parquet(self.file_path, engine='fastparquet')
        n_batches = len(parquet) // self.batch_size
        iterations = np.array_split(parquet, n_batches)
    else:
        # Read in the first row & assert format
        first_row = pd.read_csv(self.file_path, nrows=1)
        self._assert_csv(first_row)
        iterations = pd.read_csv(self.file_path, chunksize=self.batch_size)

    if to_csv is True:
        header = True  # if to_csv is True, header should written on the first batch only
        open(output_path, 'w').close()  # clean output csv file

    if verbose:  # pragma: no cover
        print('Reading data in batches:')

    # Read csv in batches
    count = 0
    final_bars = []
    cols = ['date_time', 'open', 'high', 'low', 'close', 'volume']
    for batch in iterations:
        if verbose:  # pragma: no cover
            print('Batch number:', count)

        list_bars = self._extract_bars(data=batch)

        if to_csv is True:
            pd.DataFrame(list_bars, columns=cols).to_csv(output_path, header=header, index=False, mode='a')
            header = False
        else:
            # Append to bars list
            final_bars += list_bars
        count += 1

        # Set flag to True: notify function to use cache
        self.flag = True

    if verbose:  # pragma: no cover
        print('Returning bars \n')

    # Return a DataFrame
    if final_bars:
        bars_df = pd.DataFrame(final_bars, columns=cols)
        return bars_df

    # Processed DataFrame is stored in .csv file, return None
    return None


# update imported package to deal with advanced data structure and adjust it to reas 'parquet'
ml.data_structures.base_bars.BaseBars.batch_run = new_batch_run


class TrippleBarrier(object):
    """This class is to create indicators (features) to feed ML trading algorithm.
    The content of this class was sourced from 'mlfinlab' package.
    Objectification was made in order to run this class within 'QuantConnect' platform.
    """

    def __init__(self):
        # tbd
        return

    def get_daily_vol(self, close, lookback=100):
        """
        Snippet 3.1, page 44, Daily Volatility Estimates
        Computes the daily volatility at intraday estimation points.
        In practice we want to set profit taking and stop-loss limits that are a function of the risks involved
        in a bet. Otherwise, sometimes we will be aiming too high (tao ≫ sigma_t_i,0), and sometimes too low
        (tao ≪ sigma_t_i,0 ), considering the prevailing volatility. Snippet 3.1 computes the daily volatility
        at intraday estimation points, applying a span of lookback days to an exponentially weighted moving
        standard deviation.
        See the pandas documentation for details on the pandas.Series.ewm function.
        Note: This function is used to compute dynamic thresholds for profit taking and stop loss limits.
        :param close: Closing prices
        :param lookback: lookback period to compute volatility
        :return: series of daily volatility value
        """
        # daily vol re-indexed to close
        df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
        df0 = df0[df0 > 0]
        df0 = (pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:]))

        df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily returns
        df0 = df0.ewm(span=lookback).std()
        return df0

    # Snippet 2.4, page 39, The Symmetric CUSUM Filter.
    def cusum_filter(self, raw_time_series, threshold, time_stamps=True):
        """
        Snippet 2.4, page 39, The Symmetric Dynamic/Fixed CUSUM Filter.
        The CUSUM filter is a quality-control method, designed to detect a shift in the
        mean value of a measured quantity away from a target value. The filter is set up to
        identify a sequence of upside or downside divergences from any reset level zero.
        We sample a bar t if and only if S_t >= threshold, at which point S_t is reset to 0.
        One practical aspect that makes CUSUM filters appealing is that multiple events are not
        triggered by raw_time_series hovering around a threshold level, which is a flaw suffered by popular
        market signals such as Bollinger Bands. It will require a full run of length threshold for
        raw_time_series to trigger an event.
        Once we have obtained this subset of event-driven bars, we will let the ML algorithm determine
        whether the occurrence of such events constitutes actionable intelligence.
        Below is an implementation of the Symmetric CUSUM filter.
        Note: As per the book this filter is applied to closing prices but we extended it to also work on other
        time series such as volatility.
        :param raw_time_series: (series) of close prices (or other time series, e.g. volatility).
        :param threshold: (float or pd.Series) when the abs(change) is larger than the threshold, the function captures
        it as an event, can be dynamic if threshold is pd.Series
        :param time_stamps: (bool) default is to return a DateTimeIndex, change to false to have it return a list.
        :return: (datetime index vector) vector of datetimes when the events occurred. This is used later to sample.
        """

        t_events = []
        s_pos = 0
        s_neg = 0

        # log returns
        raw_time_series = pd.DataFrame(raw_time_series)  # Convert to DataFrame
        raw_time_series.columns = ['price']
        raw_time_series['log_ret'] = raw_time_series.price.apply(np.log).diff()
        if isinstance(threshold, (float, int)):
            raw_time_series['threshold'] = threshold
        elif isinstance(threshold, pd.Series):
            raw_time_series.loc[threshold.index, 'threshold'] = threshold
        else:
            raise ValueError('threshold is neither float nor pd.Series!')

        raw_time_series = raw_time_series.iloc[1:]  # Drop first na values

        # Get event time stamps for the entire series
        for tup in raw_time_series.itertuples():
            thresh = tup.threshold
            pos = float(s_pos + tup.log_ret)
            neg = float(s_neg + tup.log_ret)
            s_pos = max(0.0, pos)
            s_neg = min(0.0, neg)

            if s_neg < -thresh:
                s_neg = 0
                t_events.append(tup.Index)

            elif s_pos > thresh:
                s_pos = 0
                t_events.append(tup.Index)

        # Return DatetimeIndex or list
        if time_stamps:
            event_timestamps = pd.DatetimeIndex(t_events)
            return event_timestamps

        return t_events

    # Snippet 3.4 page 49, Adding a Vertical Barrier
    def add_vertical_barrier(self, t_events, close, num_days=0, num_hours=0, num_minutes=0, num_seconds=0):
        """
        Snippet 3.4 page 49, Adding a Vertical Barrier
        For each index in t_events, it finds the timestamp of the next price bar at or immediately after
        a number of days num_days. This vertical barrier can be passed as an optional argument t1 in get_events.
        This function creates a series that has all the timestamps of when the vertical barrier would be reached.
        :param t_events: (series) series of events (symmetric CUSUM filter)
        :param close: (series) close prices
        :param num_days: (int) number of days to add for vertical barrier
        :param num_hours: (int) number of hours to add for vertical barrier
        :param num_minutes: (int) number of minutes to add for vertical barrier
        :param num_seconds: (int) number of seconds to add for vertical barrier
        :return: (series) timestamps of vertical barriers
        """
        timedelta = pd.Timedelta(
            '{} days, {} hours, {} minutes, {} seconds'.format(num_days, num_hours, num_minutes, num_seconds))
        # Find index to closest to vertical barrier
        nearest_index = close.index.searchsorted(t_events + timedelta)

        # Exclude indexes which are outside the range of close price index
        nearest_index = nearest_index[nearest_index < close.shape[0]]

        # Find price index closest to vertical barrier time stamp
        nearest_timestamp = close.index[nearest_index]
        filtered_events = t_events[:nearest_index.shape[0]]

        vertical_barriers = pd.Series(data=nearest_timestamp, index=filtered_events)
        return vertical_barriers

    # Snippet 20.5 (page 306), the lin_parts function
    def lin_parts(self, num_atoms, num_threads):
        """
        Snippet 20.5 (page 306), the lin_parts function
        The simplest way to form molecules is to partition a list of atoms in subsets of equal size,
        where the number of subsets is the minimum between the number of processors and the number
        of atoms. For N subsets we need to find the N+1 indices that enclose the partitions.
        This logic is demonstrated in Snippet 20.5.
        This function partitions a list of atoms in subsets (molecules) of equal size.
        An atom is a set of indivisible set of tasks.
        """
        # Partition of atoms with a single loop
        parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
        parts = np.ceil(parts).astype(int)
        return parts

    # Snippet 3.2, page 45, Triple Barrier Labeling Method
    def apply_pt_sl_on_t1(self, close, events, pt_sl, molecule):  # pragma: no cover
        """
        Snippet 3.2, page 45, Triple Barrier Labeling Method
        This function applies the triple-barrier labeling method. It works on a set of
        datetime index values (molecule). This allows the program to parallelize the processing.
        Mainly it returns a DataFrame of timestamps regarding the time when the first barriers were reached.
        :param close: (series) close prices
        :param events: (series) of indices that signify "events" (see cusum_filter function
        for more details)
        :param pt_sl: (array) element 0, indicates the profit taking level; element 1 is stop loss level
        :param molecule: (an array) a set of datetime index values for processing
        :return: DataFrame of timestamps of when first barrier was touched
        """
        # Apply stop loss/profit taking, if it takes place before t1 (end of event)
        events_ = events.loc[molecule]
        out = events_[['t1']].copy(deep=True)

        profit_taking_multiple = pt_sl[0]
        stop_loss_multiple = pt_sl[1]

        # Profit taking active
        if profit_taking_multiple > 0:
            profit_taking = profit_taking_multiple * events_['trgt']
        else:
            profit_taking = pd.Series(index=events.index)  # NaNs

        # Stop loss active
        if stop_loss_multiple > 0:
            stop_loss = -stop_loss_multiple * events_['trgt']
        else:
            stop_loss = pd.Series(index=events.index)  # NaNs

        # Get events
        for loc, vertical_barrier in events_['t1'].fillna(close.index[-1]).iteritems():
            closing_prices = close[loc: vertical_barrier]  # Path prices for a given trade
            cum_returns = (closing_prices / close[loc] - 1) * events_.at[loc, 'side']  # Path returns
            out.loc[loc, 'sl'] = cum_returns[cum_returns < stop_loss[loc]].index.min()  # Earliest stop loss date
            out.loc[loc, 'pt'] = cum_returns[
                cum_returns > profit_taking[loc]].index.min()  # Earliest profit taking date

        return out

    # Snippet 20.7 (page 310), The mpPandasObj, used at various points in the book
    def mp_pandas_obj(self, func, pd_obj, num_threads=24, mp_batches=1, lin_mols=True, **kargs):
        """
        Snippet 20.7 (page 310), The mpPandasObj, used at various points in the book
        Parallelize jobs, return a dataframe or series.
        Example: df1=mp_pandas_obj(func,('molecule',df0.index),24,**kwds)
        First, atoms are grouped into molecules, using linParts (equal number of atoms per molecule)
        or nestedParts (atoms distributed in a lower-triangular structure). When mpBatches is greater
        than 1, there will be more molecules than cores. Suppose that we divide a task into 10 molecules,
        where molecule 1 takes twice as long as the rest. If we run this process in 10 cores, 9 of the
        cores will be idle half of the runtime, waiting for the first core to process molecule 1.
        Alternatively, we could set mpBatches=10 so as to divide that task in 100 molecules. In doing so,
        every core will receive equal workload, even though the first 10 molecules take as much time as the
        next 20 molecules. In this example, the run with mpBatches=10 will take half of the time consumed by
        mpBatches=1.
        Second, we form a list of jobs. A job is a dictionary containing all the information needed to process
        a molecule, that is, the callback function, its keyword arguments, and the subset of atoms that form
        the molecule.
        Third, we will process the jobs sequentially if numThreads==1 (see Snippet 20.8), and in parallel
        otherwise (see Section 20.5.2). The reason that we want the option to run jobs sequentially is for
        debugging purposes. It is not easy to catch a bug when programs are run in multiple processors.
        Once the code is debugged, we will want to use numThreads>1.
        Fourth, we stitch together the output from every molecule into a single list, series, or dataframe.
        :param func: A callback function, which will be executed in parallel
        :param pd_obj: (tuple) Element 0: The name of the argument used to pass molecules to the callback function
                        Element 1: A list of indivisible tasks (atoms), which will be grouped into molecules
        :param num_threads: (int) The number of threads that will be used in parallel (one processor per thread)
        :param mp_batches: (int) Number of parallel batches (jobs per core)
        :param lin_mols: (bool) Tells if the method should use linear or nested partitioning
        :param kargs: (var args) Keyword arguments needed by func
        :return: (data frame) of results
        """

        if lin_mols:
            parts = self.lin_parts(len(pd_obj[1]), num_threads * mp_batches)
        else:
            print("nested parts... to fix")
            # parts = nested_parts(len(pd_obj[1]), num_threads * mp_batches)

        jobs = []
        for i in range(1, len(parts)):
            job = {pd_obj[0]: pd_obj[1][parts[i - 1]:parts[i]], 'func': func}
            job.update(kargs)
            jobs.append(job)

        if num_threads == 1:
            out = self.process_jobs_(jobs)
        else:
            out = self.process_jobs(jobs, num_threads=num_threads)

        if isinstance(out[0], pd.DataFrame):
            df0 = pd.DataFrame()
        elif isinstance(out[0], pd.Series):
            df0 = pd.Series()
        else:
            return out

        for i in out:
            df0 = df0.append(i)

        df0 = df0.sort_index()
        return df0

    # Snippet 20.8, pg 311, Single thread execution, for debugging
    def process_jobs_(self, jobs):
        """
        # Snippet 20.8, pg 311, Single thread execution, for debugging
        Run jobs sequentially, for debugging
        """
        out = []
        for job in jobs:
            out_ = self.expand_call(job)
            out.append(out_)

        return out

    # Snippet 20.9.2, pg 312, Example of Asynchronous call to pythons multiprocessing library
    def process_jobs(self, jobs, task=None, num_threads=24):
        """
        Snippet 20.9.2, pg 312, Example of Asynchronous call to pythons multiprocessing library
        Run in parallel. jobs must contain a 'func' callback, for expand_call
        """

        if task is None:
            task = jobs[0]['func'].__name__

        pool = mp.Pool(processes=num_threads)
        outputs = pool.imap_unordered(self.expand_call, jobs)
        out = []
        time0 = time.time()

        # Process asynchronous output, report progress
        for i, out_ in enumerate(outputs, 1):
            out.append(out_)
            self.report_progress(i, len(jobs), time0, task)

        pool.close()
        pool.join()  # This is needed to prevent memory leaks
        return out

    # Snippet 20.10 Passing the job (molecule) to the callback function
    def expand_call(self, kargs):
        """
        Snippet 20.10 Passing the job (molecule) to the callback function
        Expand the arguments of a callback function, kargs['func']
        """
        func = kargs['func']
        del kargs['func']
        out = func(**kargs)
        return out

    # Snippet 20.9.1, pg 312, Example of Asynchronous call to pythons multiprocessing library
    def report_progress(self, job_num, num_jobs, time0, task):
        """
        Snippet 20.9.1, pg 312, Example of Asynchronous call to pythons multiprocessing library
        """
        # Report progress as asynch jobs are completed
        msg = [float(job_num) / num_jobs, (time.time() - time0) / 60.0]
        msg.append(msg[1] * (1 / msg[0] - 1))
        time_stamp = str(dt.datetime.fromtimestamp(time.time()))

        msg = time_stamp + ' ' + str(round(msg[0] * 100, 2)) + '% ' + task + ' done after ' + str(
            round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'

        if job_num < num_jobs:
            sys.stderr.write(msg + '\r')
        else:
            sys.stderr.write(msg + '\n')

    # Snippet 3.3 -> 3.6 page 50, Getting the Time of the First Touch, with Meta Labels
    def get_events(self, close, t_events, pt_sl, target, min_ret, num_threads, vertical_barrier_times=False,
                   side_prediction=None):
        """
        Snippet 3.6 page 50, Getting the Time of the First Touch, with Meta Labels
        This function is orchestrator to meta-label the data, in conjunction with the Triple Barrier Method.
        :param close: (series) Close prices
        :param t_events: (series) of t_events. These are timestamps that will seed every triple barrier.
            These are the timestamps selected by the sampling procedures discussed in Chapter 2, Section 2.5.
            Eg: CUSUM Filter
        :param pt_sl: (2 element array) element 0, indicates the profit taking level; element 1 is stop loss level.
            A non-negative float that sets the width of the two barriers. A 0 value means that the respective
            horizontal barrier (profit taking and/or stop loss) will be disabled.
        :param target: (series) of values that are used (in conjunction with pt_sl) to determine the width
            of the barrier. In this program this is daily volatility series.
        :param min_ret: (float) The minimum target return required for running a triple barrier search.
        :param num_threads: (int) The number of threads concurrently used by the function.
        :param vertical_barrier_times: (series) A pandas series with the timestamps of the vertical barriers.
            We pass a False when we want to disable vertical barriers.
        :param side_prediction: (series) Side of the bet (long/short) as decided by the primary model
        :return: (data frame) of events
                -events.index is event's starttime
                -events['t1'] is event's endtime
                -events['trgt'] is event's target
                -events['side'] (optional) implies the algo's position side
                -events['pt'] Profit taking multiple
                -events['sl'] Stop loss multiple
        """

        # 1) Get target
        target = target.loc[t_events]
        target = target[target > min_ret]  # min_ret

        # 2) Get vertical barrier (max holding period)
        if vertical_barrier_times is False:
            vertical_barrier_times = pd.Series(pd.NaT, index=t_events)

        # 3) Form events object, apply stop loss on vertical barrier
        if side_prediction is None:
            side_ = pd.Series(1.0, index=target.index)
            pt_sl_ = [pt_sl[0], pt_sl[0]]
        else:
            side_ = side_prediction.loc[target.index]  # Subset side_prediction on target index.
            pt_sl_ = pt_sl[:2]

        # Create a new df with [v_barrier, target, side] and drop rows that are NA in target
        events = pd.concat({'t1': vertical_barrier_times, 'trgt': target, 'side': side_}, axis=1)
        events = events.dropna(subset=['trgt'])

        # Apply Triple Barrier
        first_touch_dates = self.mp_pandas_obj(func=self.apply_pt_sl_on_t1,
                                               pd_obj=('molecule', events.index),
                                               num_threads=num_threads,
                                               close=close,
                                               events=events,
                                               pt_sl=pt_sl_)

        events['t1'] = first_touch_dates.dropna(how='all').min(axis=1)  # pd.min ignores nan

        if side_prediction is None:
            events = events.drop('side', axis=1)

        # Add profit taking and stop loss multiples for vertical barrier calculations
        events['pt'] = pt_sl[0]
        events['sl'] = pt_sl[1]

        return events

    # Snippet 3.9, pg 55, Question 3.3
    def barrier_touched(self, out_df, events):
        """
        Snippet 3.9, pg 55, Question 3.3
        Adjust the getBins function (Snippet 3.7) to return a 0 whenever the vertical barrier is the one touched first.
        Top horizontal barrier: 1
        Bottom horizontal barrier: -1
        Vertical barrier: 0
        :param out_df: (DataFrame) containing the returns and target
        :param events: (DataFrame) The original events data frame. Contains the pt sl multiples needed here.
        :return: (DataFrame) containing returns, target, and labels
        """
        store = []
        for date_time, values in out_df.iterrows():
            ret = values['ret']
            target = values['trgt']

            pt_level_reached = ret > target * events.loc[date_time, 'pt']
            sl_level_reached = ret < -target * events.loc[date_time, 'sl']

            if ret > 0.0 and pt_level_reached:
                # Top barrier reached
                store.append(1)
            elif ret < 0.0 and sl_level_reached:
                # Bottom barrier reached
                store.append(-1)
            else:
                # Vertical barrier reached
                store.append(0)

        # Save to 'bin' column and return
        out_df['bin'] = store
        return out_df

    # Snippet 3.4 -> 3.7, page 51, Labeling for Side & Size with Meta Labels
    def get_bins(self, triple_barrier_events, close):
        """
        Snippet 3.7, page 51, Labeling for Side & Size with Meta Labels
        Compute event's outcome (including side information, if provided).
        events is a DataFrame where:
        Now the possible values for labels in out['bin'] are {0,1}, as opposed to whether to take the bet or pass,
        a purely binary prediction. When the predicted label the previous feasible values {−1,0,1}.
        The ML algorithm will be trained to decide is 1, we can use the probability of this secondary prediction
        to derive the size of the bet, where the side (sign) of the position has been set by the primary model.
        :param triple_barrier_events: (data frame)
                    -events.index is event's starttime
                    -events['t1'] is event's endtime
                    -events['trgt'] is event's target
                    -events['side'] (optional) implies the algo's position side
                    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
                    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
        :param close: (series) close prices
        :return: (data frame) of meta-labeled events
        """

        # 1) Align prices with their respective events
        events_ = triple_barrier_events.dropna(subset=['t1'])
        all_dates = events_.index.union(other=events_['t1'].values).drop_duplicates()
        prices = close.reindex(all_dates, method='bfill')

        # 2) Create out DataFrame
        out_df = pd.DataFrame(index=events_.index)
        # Need to take the log returns, else your results will be skewed for short positions
        out_df['ret'] = np.log(prices.loc[events_['t1'].values].values) - np.log(prices.loc[events_.index])
        out_df['trgt'] = events_['trgt']

        # Meta labeling: Events that were correct will have pos returns
        if 'side' in events_:
            out_df['ret'] = out_df['ret'] * events_['side']  # meta-labeling

        # Added code: label 0 when vertical barrier reached
        out_df = self.barrier_touched(out_df, triple_barrier_events)

        # Meta labeling: label incorrect events with a 0
        if 'side' in events_:
            out_df.loc[out_df['ret'] <= 0, 'bin'] = 0

        # Transform the log returns back to normal returns.
        out_df['ret'] = np.exp(out_df['ret']) - 1

        # Add the side to the output. This is useful for when a meta label model must be fit
        tb_cols = triple_barrier_events.columns
        if 'side' in tb_cols:
            out_df['side'] = triple_barrier_events['side']

        return out_df


def get_side(data):
    fast_window = fast
    slow_window = slow

    data['fast_mavg'] = data['close'].rolling(window=fast_window, min_periods=fast_window, center=False).mean()
    data['slow_mavg'] = data['close'].rolling(window=slow_window, min_periods=slow_window, center=False).mean()
    data.head()

    # Compute sides
    data['side'] = np.nan

    long_signals = data['fast_mavg'] >= data['slow_mavg']
    short_signals = data['fast_mavg'] < data['slow_mavg']
    data.loc[long_signals, 'side'] = 1
    data.loc[short_signals, 'side'] = -1

    # Remove Look ahead biase by lagging the signal
    data['side'] = data['side'].shift(1)

    return data


def get_indicators(raw_data):
    # Log Returns
    raw_data['log_ret'] = np.log(raw_data['close']).diff()

    # Momentum
    raw_data['mom1'] = raw_data['close'].pct_change(periods=1)
    raw_data['mom2'] = raw_data['close'].pct_change(periods=2)
    raw_data['mom3'] = raw_data['close'].pct_change(periods=3)
    raw_data['mom4'] = raw_data['close'].pct_change(periods=4)
    raw_data['mom5'] = raw_data['close'].pct_change(periods=5)

    # Volatility
    raw_data['volatility_50'] = raw_data['log_ret'].rolling(window=50, min_periods=50, center=False).std()
    raw_data['volatility_31'] = raw_data['log_ret'].rolling(window=31, min_periods=31, center=False).std()
    raw_data['volatility_15'] = raw_data['log_ret'].rolling(window=15, min_periods=15, center=False).std()

    # Serial Correlation (Takes about 4 minutes)
    # GBM data is lack of serial correlation, thus disabled

    window_autocorr = 50

    raw_data['autocorr_1'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr,
                                                         center=False).apply(lambda x: x.autocorr(lag=1), raw=False)
    raw_data['autocorr_2'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr,
                                                         center=False).apply(lambda x: x.autocorr(lag=2), raw=False)
    raw_data['autocorr_3'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr,
                                                         center=False).apply(lambda x: x.autocorr(lag=3), raw=False)
    raw_data['autocorr_4'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr,
                                                         center=False).apply(lambda x: x.autocorr(lag=4), raw=False)
    raw_data['autocorr_5'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr,
                                                         center=False).apply(lambda x: x.autocorr(lag=5), raw=False)

    # Get the various log -t returns
    raw_data['log_t1'] = raw_data['log_ret'].shift(1)
    raw_data['log_t2'] = raw_data['log_ret'].shift(2)
    raw_data['log_t3'] = raw_data['log_ret'].shift(3)
    raw_data['log_t4'] = raw_data['log_ret'].shift(4)
    raw_data['log_t5'] = raw_data['log_ret'].shift(5)

    # Re compute sides
    raw_data['side'] = np.nan

    long_signals = raw_data['fast_mavg'] >= raw_data['slow_mavg']
    short_signals = raw_data['fast_mavg'] < raw_data['slow_mavg']

    raw_data.loc[long_signals, 'side'] = 1
    raw_data.loc[short_signals, 'side'] = -1

    # Remove look ahead bias
    raw_data = raw_data.shift(1)

    return raw_data


# source folders
my_dir = os.getcwd()
ticks_folder = os.path.join(my_dir, "data/5_AdjTicks")

# destination folder / path to files with dollar bars
dollar_bars_folder = os.path.join(my_dir, "data/6_DollarBars")
if os.path.basename(dollar_bars_folder) not in os.listdir(os.path.dirname(dollar_bars_folder)):
    os.mkdir(dollar_bars_folder)

# destination folder / path to files with indicators
# indicators_folder = os.path.expanduser('~/Downloads/Indicators')
indicators_folder = os.path.join(my_dir, "data/7_Indicators")
if os.path.basename(indicators_folder) not in os.listdir(os.path.dirname(indicators_folder)):
    os.mkdir(indicators_folder)

keys = [key[:4] for key in os.listdir(ticks_folder) if not key.startswith(".")]
print(keys)

# Input parameters

est_ticks = 10  # per day

vertical_barrier_days = 5  # days

# the following parameters need to be adjusted for particular case
pt_sl = [1, 2]
min_ret = 1 / 100  # triple_barrier_boundary

# sma
fast = 20
slow = 50

get_dollar_bars_file_name = lambda key, est_ticks: f"{key}_{str(est_ticks)}_dollar_bars.csv"

for key in keys:

    ticks_file = [f for f in os.listdir(ticks_folder) if key in f][0]
    ticks_file_path = os.path.join(ticks_folder, ticks_file)

    dollar_bars_path = os.path.join(dollar_bars_folder, get_dollar_bars_file_name(key, est_ticks))

    if os.path.basename(dollar_bars_path) not in os.listdir(os.path.dirname(dollar_bars_path)):
        # indicators_path = os.path.join(indicators_folder, (key+ '_indicators.csv'))

        # Select DollarBar size
        ticks = pd.read_parquet(ticks_file_path)

        # In[6]:

        # overall traded volume
        N = ticks[['price', 'volume']].prod(axis=1).sum()

        # number of days traded
        D = np.unique(ticks.date_time.values.astype('M8[D]')).shape[0]

        # estimated threshold wrt estimated dayly amount of ticks
        threshold = np.round((N / D) / est_ticks)
        print('Creating Dollar Bars for ', key)
        print("N of ticks: ", N, "trading days: ", D, "dollars of trade in dollar bar: ", threshold)

        # ## Create dollar bars
        dollar = ml.data_structures.get_dollar_bars(ticks_file_path,
                                                    threshold=threshold, batch_size=5000000,
                                                    verbose=True, to_csv=True,
                                                    output_path=dollar_bars_path)

get_indicators_name = lambda key, vbd, minret: f"{key}_{str(vbd)}_{str(minret * 100)}_indicators.csv"

for key in keys:
    dollar_bars_path = os.path.join(dollar_bars_folder, get_dollar_bars_file_name(key, est_ticks))
    indicators_path = os.path.join(indicators_folder,
                                   get_indicators_name(key, vertical_barrier_days, min_ret))

    if os.path.basename(indicators_path) not in os.listdir(os.path.dirname(indicators_path)):
        data = pd.read_csv(dollar_bars_path, index_col=0, parse_dates=True)
        print("data shape for ", key, " - ", data.shape)

        # data heads: ['open', 'high', 'low', 'close'] ?? cum_vol    cum_dollar    cum_ticks
        ############ get indicators:###########################################
        data = get_side(data)

        ################## build bins ###################################
        # Save the raw data
        raw_data = data.copy()

        # Drop the NaN values from our data set
        data.dropna(axis=0, how='any', inplace=True)

        trplbr = TrippleBarrier()

        # Compute daily volatility
        daily_vol = trplbr.get_daily_vol(close=data['close'], lookback=50)

        # Apply Symmetric CUSUM Filter and get timestamps for events
        # Note: Only the CUSUM filter needs a point estimate for volatility
        cusum_events = trplbr.cusum_filter(data['close'], threshold=daily_vol.mean() * 0.5)

        # Compute vertical barrier
        vertical_barriers = trplbr.add_vertical_barrier(t_events=cusum_events, close=data['close'],
                                                        num_days=vertical_barrier_days)

        # the following parameters need to be adjusted for particular case
        # pt_sl = [1, 2]
        # min_ret = 0.0005
        triple_barrier_events = trplbr.get_events(close=data['close'],
                                                  t_events=cusum_events,
                                                  pt_sl=pt_sl,
                                                  target=daily_vol,
                                                  min_ret=min_ret,
                                                  num_threads=3,
                                                  vertical_barrier_times=vertical_barriers,
                                                  side_prediction=data['side'])

        # labels = ml.labeling.get_bins(triple_barrier_events, data['close'])
        labels = trplbr.get_bins(triple_barrier_events, data['close'])

        print("shape of labels :", labels.shape)

        ###################### get other indicators ####################################

        raw_data = get_indicators(raw_data)

        #### Now get the data at the specified events

        df = pd.concat([raw_data, labels], axis=1, sort=False)

        df[~df.slow_mavg.isna()].to_csv(indicators_path)

