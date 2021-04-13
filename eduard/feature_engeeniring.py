from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from featuretools.primitives import Week, Year, Weekday, Month

class Interpolation(BaseEstimator, TransformerMixin):
    def __init__(self, kind_of_interpolation='linear', astype='float32'):
        self.kind_of_interpolation = kind_of_interpolation
        self.astype = astype

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy = X_copy.interpolate(self.kind_of_interpolation).astype(self.astype)

        return X_copy


class TranformTimeSeries(BaseEstimator, TransformerMixin):

    def __init__(self, box_cox=[]):  # , diff_model=dict()):
        #         self.diff_model = diff_model
        self.box_cox = box_cox

    def fit(self, X, y=None):

        if len(self.box_cox) == 0:
            return self

        lmbds = []
        columns_lmbds = self.box_cox.copy()

        # expliciti imputing zeros value as a const
        resulting = X.copy()
        resulting[columns_lmbds] = np.where(resulting[columns_lmbds] == 0, 1e-1, resulting[columns_lmbds])

        df_box = resulting.copy()

        for column in columns_lmbds:
            print(f'boxcox column:\t{column}')
            data_box_cox, lmbd = boxcox(df_box[column])
            df_box[column] = data_box_cox # np.power(df_box[column], lmbd) # data_box_cox
            lmbds.append(lmbd)

        self.lmbds = dict((col, lmbd) for col, lmbd in zip(columns_lmbds, lmbds))

        return self

    def transform(self, X, y=None):

        resulting = X.copy()
        resulting[self.box_cox] = np.where(resulting[self.box_cox] == 0, 1e-1, resulting[self.box_cox])

        df_box = resulting.copy()

        for column, lmbd in self.lmbds.items():
            #             print(f'column:\t{column}')
            data_box_cox = boxcox(df_box[column], lmbda=lmbd)
            df_box[column] = data_box_cox# np.power(df_box[column], lmbd)

        return df_box


class Lags(BaseEstimator, TransformerMixin):
    def __init__(self, lags=0, columns=[]):
        self.lags = lags
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        transformed = pd.DataFrame()

        # creates lags for all features
        for i in range(1, self.lags + 1):
            col_my_new = list(x + '-1' for x in self.columns)
            transformed[col_my_new] = X_copy[self.columns].shift(i)

        result = pd.concat([X_copy, transformed], axis=1)
        return result.dropna()


class RollingMeanAverage(BaseEstimator, TransformerMixin):
    def __init__(self, window=2, columns=[]):
        self.window = window
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        transformed = pd.DataFrame()

        # creates rolling mean and std
        for i in range(2, self.window + 1):
            col_mean = list(x + f'_av_mean_{i}' for x in self.columns)

            transformed[col_mean] = X_copy[self.columns].rolling(window=i).mean()

        result = pd.concat([X_copy, transformed], axis=1)
        return result.dropna()


class RollingStdAverage(BaseEstimator, TransformerMixin):
    def __init__(self, window=2, columns=[]):
        self.window = window
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        transformed = pd.DataFrame()

        # creates rolling mean and std
        for i in range(2, self.window + 1):
            col_std = list(x + f'_av_std_{i}' for x in self.columns)
            transformed[col_std] = X_copy[self.columns].rolling(window=i).std()

        result = pd.concat([X_copy, transformed], axis=1)
        return result.dropna()


class RollingMeanExponential(BaseEstimator, TransformerMixin):
    def __init__(self, window=2, com=0.8, columns=[]):
        self.window = window
        self.com = com
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        transformed = pd.DataFrame()

        # creates rolling mean and std
        for i in range(2, self.window + 1):
            col_mean = list(x + f'_ewm_mean_{i}' for x in self.columns)
            transformed[col_mean] = X_copy[self.columns].ewm(com=self.com).mean()

        result = pd.concat([X_copy, transformed], axis=1)
        return result.dropna()


class RollingStdExponential(BaseEstimator, TransformerMixin):
    def __init__(self, window=2, com=0.8, columns=[]):
        self.window = window
        self.com = com
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        transformed = pd.DataFrame()

        # creates rolling mean and std
        for i in range(2, self.window + 1):
            col_std = list(x + f'_ewn_std_{i}' for x in self.columns)
            transformed[col_std] = X_copy[self.columns].ewm(com=self.com).std()

        result = pd.concat([X_copy, transformed], axis=1)
        return result.dropna()


from featuretools.primitives import Week, Weekday, Year, Month


class DayWeekYear(BaseEstimator, TransformerMixin):
    def __init__(self, weak=True, month=True, year=True, weekday=True):
        self.weak_bool = weak
        self.month_bool = month
        self.year_bool = year
        self.weekday_bool = weekday

    def fit(self, X, y=None):
        if self.weak_bool:
            self.weak = Week()

        if self.month_bool:
            self.month = Month()

        if self.year_bool:
            self.year = Year()

        if self.weekday_bool:
            self.weekday = Weekday()

        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        transformed = pd.DataFrame()

        dates = X_copy.index

        if self.weak_bool:
            X_copy['weak'] = self.weak(dates).tolist()

        if self.month_bool:
            X_copy['month'] = self.month(dates).tolist()

        if self.year_bool:
            X_copy['year'] = self.year(dates).tolist()

        if self.weekday_bool:
            X_copy['weekDay'] = self.weekday(dates).tolist()

        return X_copy


class DinamicsForTimeSeries(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[], dropna=True):
        self.columns = columns
        self.dropna = dropna

    def fit(self, X, y=None):
        X_copy = X.copy()
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_shift = X_copy[self.columns].shift(1)
        X_dim = X_copy[self.columns] / X_shift
        new_columns = list(x + '_dinamic' for x in self.columns)
        X_dim.columns = new_columns

        X_with_dinamic = pd.concat([X_copy, X_dim], axis=1)

        if self.dropna:
            X_with_dinamic = X_with_dinamic.dropna()

        return X_with_dinamic


if __name__ == '__main__':
    pass