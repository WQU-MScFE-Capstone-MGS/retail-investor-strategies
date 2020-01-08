# Tick Data Strategy

To proceed one needs to download and to preprocess data for feeding algorithm. Below is the file / data structure and intstructions how to run scripts.

**folders/files structure:**
<pre>
data
|__ 1_RawTicks
    |__ ...
|__ 2_MOEX
    |__ ...
|__ 3_Dividends
    |__ ...
|__ 4_Ticks
    |__ ...
|__ 5_AdjTicks
    |__ ...
|__ 6_DollarBars
    |__ ...
|__ 7_Indicators
    |__ ...
1_get_tick_data.py
2_preprocess_ticks.py
3_create_adj_ticks.py
4_dollar_bars_triple_barrier_indicators.py
QCTickDataStrategy.py
</pre>


To create python environment one needs to use conda and use 'requirements.txt'. This environment should be built on python verios 3.6.

<code>conda create --name QC python=3.6.6</code>

<code>conda activate QC</code>

<code>pip install -r requirements.txt</code>

It will be used for running python scripts as well as for LEAN Engine.


## Data Download

This script was tested on Ubuntu 18.04 and MacOs (Mojave, Catalina) with FireFox browser.

To download the data one required to have selenium webdrive installed. Instructions for that could be found [here](https://selenium-python.readthedocs.io/installation.html#drivers)

Among blue chips and MOEX index constituents the following assets were chosen for the analysis:
'AFKS', 'ALRS', 'CHMF', 'GAZP', 'GMKN', 'LKOH', 'MGNT', 'MTSS', 'NVTK', 'ROSN', 'RTKM', 'SBER', 'SNGS', 'TATN', 'VRBR', 'YNDX'

To start downloading process one needs to run '1_get_tick_data.py' from command line with the arguments 'symbol', start date and end date 'YYYY-MM-DD', i.e.:

<code>python.py 1_get_tick_data.py GAZP 2009-01-01 2019-12-13</code>


After the algorithm opens the firefox window, the frequency of the data ('ticks') and the output format ('.csv') need to be selected manually. In addition, one needs to select 'save to file' and select a checkbox 'repeat for the next occurencies'. All this need to be done onces.

Algorithm will dowload chunks of data in .csv files withing size limit of around 41.6Mb into folders named by symbol into '1_RawTicks' folder.

This web-site _does not_ provide data to download every day from 7:00am to 3:00pm GMT.

Alternatavly, raw tick data for this research can be reached in [Box folder](https://app.box.com/s/fwau5uwsrvn4lgwfwpvkf9zwnxo24k82)


## Data Preprocess

These scripts preprocess raw tick data to create dollarbars and indicators (features) for feedin ML algorithm.

1. Run 2_preprocess_ticks.py to save data within single parquet file in the folder '2_Ticks' for each company.

2. Adjust tick data backward to dividends paid - see [formula](https://help.yahoo.com/kb/SLN28256.html). Dividends data is manually downloaded from 'finance.yahoo.com' and saved to the folder '3_Dividends'. Run 3_create_adj_ticks.py for adjusting and saving data into the folder '3_AdjTicks'.

3. The last preprocessing script 4_dollar_bars_triple_barrier_indicators.py creates dollarbars and save them into folder '4_DollarBars', then creates dataseries to feed trading altorithm and saves them into folder '5_Indicators'. The dollarbars and indicators are built based on input parameters, that could be changed for modelling variations.


## Running Algos

Trading algorithm is developed to run within open source QuantConnect platform. Trading algorithm could be executed on the web [QuantConnect](https://www.quantconnect.com) service or locally on the underlying [LEAN Engine](https://github.com/QuantConnect/Lean/tree/master/Algorithm.Python#quantconnect-python-algorithm-project).

- To run it with QuantConnect platform one needs to login, 'Create new Algorithm' within 'Algorithm Lab' (or 'Lab'), and substitute the code by the code from 'QCTickDataStrategy.py'. This script contains links to dropbox folders with already created 'Indicators' files, which can be substituted by the files you created at the 'Data Preprocess' step above. After starting 'Backtest', QuantConnect will generate statistics and reports. (This scripts were tested under PRO account and could be running slowly under free account).

- To run algorithm locally, one needs to have Visual Studio and python envrionment, which was created at the first step. Details on istallation, compiling and running algotithm are available [here](https://medium.com/hackernoon/setting-up-your-own-algorithmic-trading-server-4bbdf0766c17). In this step the dropbox links to files with indidators can be substituted to local file links.

