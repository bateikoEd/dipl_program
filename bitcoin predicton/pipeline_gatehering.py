import pandas as pd
from datetime import date
from pytrends.request import TrendReq
import yahoofinance as yf
from tqdm import tqdm

class GatheringData():

    def __init__(self, start_day='2016-04-10', end_day=None):
        self.df_pytrends = None
        self.df_yahoo = None
        self.start_day = start_day
        self.end_day = end_day
        self.pytrend = TrendReq()


    def get_from_google_trends(self, kw_list=[],
                               timeframe='today 5-y',
                               diretory=None,
                               resample=None):

        assert len(kw_list) > 0, f'list of words must be not 0'

        resulting_df = []

        print('gathering pytrends')
        for word in tqdm(kw_list):
            word_l = [word]
            get_result = self.pytrend.build_payload(word_l, cat=0, timeframe=timeframe, geo='', gprop='')
            result = self.pytrend.interest_over_time().drop('isPartial', axis=1)

            resulting_df.append(result)

            if diretory is not None:
                result.to_csv(f'{diretory}_{timeframe}.csv')


        if len(kw_list) > 1:
            self.df_pytrends = pd.concat(resulting_df, axis=1)
        else:
            self.df_pytrends = resulting_df[0]

        if resample is not None:
            self.df_pytrends = self.df_pytrends.resample(resample).mean()

        return self.df_pytrends.copy()


    def get_from_yahoo(self, kw_list=[]):
        assert len(kw_list) > 0, f'list of words must be not 0'

        if self.end_day is None:
            self.end_day = date.today()

        #         self.start_day = str(self.df_pytrends.index[0])[:6]
        data_queries = []

        kw_len = len(kw_list)

        print('gathering yahoo')

        for word in tqdm(kw_list):
            data = yf.HistoricalPrices(word, start_date=self.start_day, end_date=self.end_day).to_dfs()
            data = data['Historical Prices']

            if kw_len >= 2:
                data = pd.DataFrame(data['Close'])
                data.columns = [f'Close_{word}']

            data_queries.append(data)

        if kw_len < 2:
            self.df_yahoo = pd.concat(data_queries, axis=1)

        else:
            kw_list = [x + '_close' for x in kw_list]
            self.df_yahoo = pd.concat(data_queries, names=kw_list, axis=1)

        return self.df_yahoo.copy()


    def get_resulting_from_sources(self):

        self.df_pytrends.index = self.df_yahoo.index[:self.df_pytrends.shape[0]]

        self.df_concated_yahoo_pytrends = pd.concat([self.df_yahoo, self.df_pytrends], axis=1).astype('float32')
        return self.df_concated_yahoo_pytrends.copy()


    def drop_columns_result_data(self, columns=[]):
        assert len(columns) > 0, 'count of columns need to be not 0'

        self.df_concated_yahoo_pytrends = self.df_concated_yahoo_pytrends.drop(columns, axis=1)

        return self.df_concated_yahoo_pytrends.copy()


    def to_csv_df_concated_yahoo_pytrends(self, path=None):
        assert path is None, 'path does not be None'

        self.df_concated_yahoo_pytrends.to_csv(path)
        return self


    def to_csv_df_yahoo(self, path=None):
        assert path is None, 'path does not be None'

        self.df_yahoo.to_csv(path)
        return self


    def to_csv_df_pytrends(self, path=None):
        assert path is None, 'path does not be None'

        self.df_pytrends.to_csv(path)
        return self

if __name__ == '__main__':
    pass