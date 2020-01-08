# bash python==3
# Project to download tick data (MOEX) from finam.ru

import os
import sys
import time
import math
import datetime as dt

# 'selenuim' package for manipulation with the website
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains

url = 'https://www.finam.ru/profile/moex-akcii/gazprom-neft/export/?market=1&em=2&code=SIBN&apply=0&df=1&mf=0&' +\
      'yf=2017&from=01.01.2017&dt=31&mt=0&yt=2017&to=31.01.2017&p=1&f=SIBN_170101_170131&e=.csv&cn=SIBN&dtf=1&' +\
      'tmf=1&MSOR=1&mstime=on&mstimever=1&sep=1&sep2=1&datf=6&at=1'


class GetTickData(object):
    """A class to download data from finam.ru by chanks wrt restriction to downloaded file size ~41k
    """

    def __init__(self, name, folder=None, url=url):
        """initiation of the selenium and download process
        """

        self.name = name
        if folder is None:
            folder = os.getcwd()
        self.parent_folder = folder
        self.folder = self.set_folder(folder)
        self.driver = self._get_driver()
        self.driver.get(url)
        self.wait = WebDriverWait(self.driver, 10)
        self.last_time = dt.datetime.now()
        self.out = 'csv'
        self.previous_from = dt.datetime.now().date()
        self.previous_to = dt.datetime.now().date()
        self.file_size = 40 * 2 ** 20
        self.target_days = 15
        self.files = []
        self.os = os.name

    def set_folder(self, folder):
        """Creating a folder to store the data"""
        folder_target = os.path.join(folder, self.name)
        if not (self.name in os.listdir(folder)):
            os.mkdir(folder_target)
        print("Target folder:", folder_target)
        return folder_target

    def _get_driver(self):
        """Starting the web-driver for Firefox for further manipulations with the website
        """
        mime_csv = 'text/plain, application/csv, application/download,' + \
                   ' text/comma-separated-values, text/csv, text/anytext,' + \
                   ' application/csv, application/excel,' + \
                   ' application/vnd.msexce, application/vnd.ms-excel,' + \
                   ' attachment/csv, text/plain'

        fp = webdriver.FirefoxProfile()
        fp.set_preference('browser.download.folderList', 2)
        fp.set_preference('browser.download.manager.showWhenStarting', False)
        fp.set_preference('browser.download.dir', self.folder)
        fp.set_preference('browser.helperApps.neverAsk.saveToDisk', mime_csv)
        driver = webdriver.Firefox(firefox_profile=fp)
        return driver

    def set_company(self, name):
        """Selecting a company which data will be downloaded
        """
        company = '/html/body/div[3]/div[2]/div[1]/div/table/tbody/tr/td/div/div/div[2]/div[1]/div[2]/input'

        try:
            s3 = self.driver.find_element_by_xpath(company)
            s3.clear()
            s3.send_keys(name)
            s3.send_keys(Keys.ENTER)
        except Exception as e:
            print('Error setting a company ', name, ', ', e)
            return False
        time.sleep(40)
        return True

    @staticmethod
    def _get_date_position(date):
        """function to get year, month and day positions within drop-down
        calendar
        """

        day_col = date.weekday() + 1

        # get the No of week (week starts from monday)
        first_day = date.replace(day=1)
        day = date.day
        adjusted_dom = day + first_day.weekday()
        day_ind = int(math.ceil(adjusted_dom / 7.0))  # np.ceil(a).astype(int)[0]

        return day_col, day_ind

    def set_date(self, date_from, date_to, from_or_to='from'):
        """Function to set specific date in the drop-down calendar
        """
        if from_or_to == 'from':
            date = date_from
            year_num = date.year + (40 - 2018)
            month_num = date.month
            date_sel = '#issuer-profile-export-from'
            previous = self.previous_from
        elif from_or_to == 'to':
            date = date_to
            year_num = date.year + (1 - date_from.year)
            month_num = (date.month + 1 - date_from.month) if date_from.year == date.year else date.month
            date_sel = '#issuer-profile-export-to'
            previous = self.previous_to

        day_col, day_ind = self._get_date_position(date)

        year_sel = '.ui-datepicker-year'
        month_sel = '.ui-datepicker-month'

        year = '/html/body/div[16]/div/div/select[2]/option[' + str(year_num) + ']'  # 40 - 2018
        month = '/html/body/div[16]/div/div/select[1]/option[' + str(month_num) + ']'
        day = '/html/body/div[16]/table/tbody/tr[' + str(day_ind) + ']/td[' + str(day_col) + ']/a'

        try:
            select_calendar = self.wait.until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, date_sel)))
            select_calendar.click()
        except Exception as e:
            print('error date ', from_or_to, e)
            return False

        if not previous.year == date.year:
            try:
                select_year = self.wait.until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, year_sel)))
                select_year.click()

                select_year = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, year)))

                ActionChains(self.driver).move_to_element(select_year).perform()

                self.driver.find_element_by_xpath(year).click()

            except Exception as e:
                print('error year ', from_or_to, e)
                return False

        if not previous.month == date.month:
            try:
                select_month = self.wait.until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, month_sel)))
                select_month.click()

                select_month = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, month)))
                ActionChains(self.driver).move_to_element(select_month).perform()
                self.driver.find_element_by_xpath(month).click()
            except Exception as e:
                print('error month', from_or_to, e)
                return False

        try:
            first_available_date = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, day)))
            ActionChains(self.driver).move_to_element(first_available_date).perform()
            self.driver.find_element_by_xpath(day).click()
        except Exception as e:
            print('error day', from_or_to, e)
            return False

        time.sleep(5)
        return True

    def get_data(self, date_from, date_to, chronological=False):
        """Function to download data between two dates. Remember the calendar
        validation restriction with regard to current state (thus, one cannt
        set date_to > date_from)
        """
        if not chronological:
            self.set_date(date_from, date_to, from_or_to='from')
            self.set_date(date_from, date_to, from_or_to='to')
        else:
            self.set_date(date_from, date_to, from_or_to='to')
            self.set_date(date_from, date_to, from_or_to='from')

        dnlwdxp = '/html/body/div[3]/div[2]/div[1]/div/table/tbody/tr/td/div/div/div[2]/div[2]/div/div[2]/' +\
                  'div[1]/form/div/button/span'

        try:
            downld = self.wait.until(
                EC.visibility_of_element_located((By.XPATH, dnlwdxp)))
            downld.click()

            # handle FireFox allert regarding sending sensitive date
            self.driver.switch_to.alert.accept()
        except Exception as e:
            print('Error when downloading file', e)
            return False

        print(f'Downloading data file from {date_from} to {date_to}')

        return True

    def get_file_size(self, date_from, date_to):
        """Function to check that the size of the file is not beyond the boundary ~40k and that it has data
        """
        name_key = date_from.strftime('%y%m%d') + '_' + date_to.strftime('%y%m%d') + '.' + self.out
        name = None

        i = 1
        while i < 40:
            print('Waiting for file to proceed...')
            time.sleep(15)
            list_files = os.listdir(self.folder)
            for name in list_files:
                if name_key in name:
                    i = 100
                    break
            i += 1

        if i >= 100 and (name is not None):
            file_path = os.path.join(self.folder, name)
            file_size = os.stat(file_path).st_size
        else:
            file_path = ''
            file_size = 500000

        return file_path, file_size

    def get_downloaded_date_from(self):
        """Get the earliest data of downloaded asset to continue download from then
        """
        date = dt.datetime.now().date()
        for file in os.listdir(self.folder):
            try:
                temp_date = dt.datetime.strptime(file[-17:-11], '%y%m%d').date()
            except Exception as e:
                print("file without data in the name, ", e)
                temp_date = date
            if temp_date < date:
                date = temp_date

        return date

    def get_bulk_data(self, date_from="2019-11-01", date_to="2019-11-30"):
        """Method to continuously operate downloading
        """

        date_from = dt.datetime.strptime(date_from, '%Y-%m-%d').date()
        date_to = dt.datetime.strptime(date_to, '%Y-%m-%d').date()

        downloaded_date_from = self.get_downloaded_date_from()

        if downloaded_date_from <= date_from:
            return
        else:
            if downloaded_date_from < date_to:
                date_to = downloaded_date_from - dt.timedelta(days=1)

        # set temp dates to go through daterange in reversed chronological order
        temp_date_to = date_to
        temp_date_from = temp_date_to - dt.timedelta(days=self.target_days)
        if date_from > temp_date_from:
            temp_date_from = date_from

        while date_from < temp_date_to:
            self.files = os.listdir(self.folder)
            self.get_data(temp_date_from, temp_date_to)

            expected_file_name = (self.name + "_" + dt.datetime.strftime(temp_date_from, "%y%m%d") + "_" +
                                  dt.datetime.strftime(temp_date_to, "%y%m%d") + ".csv")

            # waiting for the downloading to complete
            stop_time = dt.datetime.now() + dt.timedelta(minutes=30)
            now = dt.datetime.now()
            while now < stop_time:
                time.sleep(20)
                list_files = os.listdir(self.folder)

                if (list_files != self.files) and (expected_file_name in list_files):
                    now += dt.timedelta(minutes=30)
                    for name in list_files:
                        if ".part" in name:
                            now = dt.datetime.now()
                            print("Waiting for downloading to complete, ", (stop_time - now), " before cancel")
                else:
                    now = dt.datetime.now()
                    print("Waiting for downloading to begin, ", (stop_time - now), " before cancel")

            file_path, file_size = self.get_file_size(temp_date_from, temp_date_to)

            if file_size < 200:
                os.remove(file_path)
                raise Exception("No data provided")

            print("Finished at ", dt.datetime.now(), ", file size (b): ", file_size, )

            self.previous_to, self.previous_from = temp_date_to, temp_date_from

            # check if file_size close to the limit 40k
            if file_size < (self.file_size * 0.9):
                # self-adjusting period of download within size-limits
                self.target_days = int((self.file_size / 2) * self.target_days / file_size)
                temp_date_to = temp_date_from - dt.timedelta(days=1)
                temp_date_from = temp_date_to - dt.timedelta(days=self.target_days)
            else:
                file_name = os.path.basename(file_path)
                print(f"File {file_name} beyond the file size, retrying")
                os.remove(file_path)
                self.target_days = self.target_days // 2
                temp_date_from = temp_date_to - dt.timedelta(days=self.target_days)

        return

    def close(self):
        """closing the session of seleniun
        """
        self.driver.close()


# blue chips as of 2014 (from wikipedia history)
# finam.ru abbr. / finam.ru name
BLUE_CHIPS_DICT = dict(GAZP='ГАЗПРОМ ао', SBER='Сбербанк', LKOH='ЛУКОЙЛ', MGNT='Магнит ао', SNGS='Сургнфгз',
                       NVTK='Новатэк ао', MTSS='МТС-ао', ROSN='Роснефть', GMKN='ГМКНорНик', VTBR='ВТБ ао',
                       TATN='Татнфт 3ао', AFKS='Система ао', RTKM='Ростел -ао', YNDX='', CHMF='', ALRS='')


def main():
    my_dir = os.getcwd()
    folder = os.path.join(my_dir, "data/1_RawTicks")

    if os.path.basename(folder) not in os.listdir(os.path.dirname(folder)):
        os.mkdir(folder)

    if len(sys.argv) > 1:
        key = sys.argv[1]
    else:
        key = 'GAZP'

    if len(sys.argv) > 2:
        date_from = sys.argv[2]
    else:
        date_from = "2009-01-01"

    if len(sys.argv) > 3:
        date_to = sys.argv[3]
    else:
        date_to = "2019-12-13"

    name = BLUE_CHIPS_DICT[key]

    asset = GetTickData(key, folder=folder)

    try:
        asset.set_company(name)
        asset.get_bulk_data(date_from, date_to)
    except Exception as e:
        print('company ', key, ' failed because of ', e)

    asset.close()


if __name__ == "__main__":
    main()
