from sklearn.metrics import r2_score
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import explained_variance_score
import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
import warnings
plt.rcParams['figure.figsize'] = [20,8]
warnings.filterwarnings('ignore')


from statsmodels.graphics.gofplots import qqplot
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from sklearn.metrics import r2_score
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import explained_variance_score
import numpy as np
import pandas as pd
from scipy.special import inv_boxcox


def define_metrics(y, predicted_train, predicted_test, name):
    pred_train_ = predicted_train
    pred_test_ = predicted_test
    y_train = y[:len(predicted_train)]
    y_test = y[-len(predicted_test):]

    metric_train = pd.DataFrame()
    metric_train['name'] = [name + '_train']
    metric_train['R2'] = [r2_score(y_train, pred_train_)]
    metric_train['RSS'] = np.sum((y_train - pred_train_) ** 2)
    metric_train['MAPE'] = [np.mean(np.abs((y_train - pred_train_) / y_train)) * 100]
    metric_train['RMSE'] = [np.sqrt(np.mean((y_train - pred_train_) ** 2))]
    metric_train['durbin_watson'] = [durbin_watson(y_train - pred_train_)]
    metric_train['theil_index'] = [np.sqrt((1 / len(pred_train_)) * np.sum((y_train - pred_train_) ** 2))
                                   / (np.sqrt((1 / len(y_train)) * np.sum(y_train ** 2)) + np.sqrt(
        (1 / len(pred_train_)) * np.sum(pred_train_ ** 2)))]

    metric_train['ex_var'] = [explained_variance_score(y_train, pred_train_)]

    metric_test = pd.DataFrame()
    metric_test['name'] = [name + '_test']
    metric_test['R2'] = [r2_score(y_test, pred_test_)]
    metric_test['RSS'] = np.sum((y_test - pred_test_) ** 2)

    metric_test['MAPE'] = [np.mean(np.abs((y_test - pred_test_) / y_test)) * 100]

    metric_test['RMSE'] = [np.sqrt(np.mean((y_test - pred_test_) ** 2))]
    metric_test['durbin_watson'] = [durbin_watson(y_test - pred_test_)]
    metric_test['theil_index'] = [np.sqrt((1 / len(pred_test_)) * np.sum((y_test - pred_test_) ** 2))
                                  / (np.sqrt((1 / len(y_test)) * np.sum(y_test ** 2)) + np.sqrt(
        (1 / len(pred_test_)) * np.sum(pred_test_ ** 2)))]

    metric_test['ex_var'] = [explained_variance_score(y_test, pred_test_)]

    return metric_train.append(metric_test)
def tsplot(y, lags=None, figsize=(15, 13), style='bmh', title='', path=None):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    path_im = None
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        # mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title(title)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()

        if path is not None:
            path_im = f'{path}.png'
            plt.savefig(path_im)

    plt.close()
    return path_im


def show_pred(y_train, y_test, y, title, directory=None, file_name=None):
    #     y_train_copy = pd.Series(y_train, index=y.index[:len(y_train)])
    #     y_test_copy = pd.Series(y_test, index=y.index[-len(y_test):])
    y_test_copy = pd.Series(y_test)
    y_train_copy = pd.Series(y_train)
    #     print(y_test_copy)
    #     print(f'y_train_copy shape:\t{y_train_copy.shape}')
    #     print(f'y_test_copy shape:\t{y_test_copy.shape}')
    #     print(f'y shape:\t{y.shape}')

    plt.plot(y_train_copy, label='y_train_predicted')
    plt.plot(y_test_copy, label='y_test_predicted')
    plt.plot(y, label='y_real')
    plt.title(title)
    plt.legend()

    if directory is not None:

        path_im = f'{directory}/{file_name}.png'
        file_name_im = f'{file_name}.png'
        plt.savefig(path_im)


    plt.show()

    return path_im, file_name_im

def get_prediction_error_from_pipeline_autoMl(pipeline, X_train, X_test, y_train, y_test):
    prediction_test = pipeline.predict(X_test).to_series()
    prediction_train = pipeline.predict(X_train).to_series()

    prediction_train.index = y_train.index
    prediction_test.index = y_test.index

    error_test = prediction_test - y_test.values
    error_train = prediction_train - y_train.values

    return prediction_train, prediction_test, error_train, error_test

def show_train_test_residual(error_train, error_test, directory=None, file_name=None):
    if file_name is not None:
        print(file_name)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1 = error_train.plot(kind='kde', figsize=(15, 15), label='residual_train')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2 = error_test.plot(kind='kde', figsize=(15, 15), label='residual_test')

    # Save the full figure...
    # fig.savefig('full_figure.png')

    if directory is not None:
        path_pdf = f'{directory}/pdf_{file_name}.png'
        plt.savefig(path_pdf)
        plt.close()

    # plt.subplot(221)
    # error_train.plot(kind='kde', figsize=(15, 15), label='residual_train')
    # plt.title('residual_train')
    #
    # plt.subplot(222)
    # error_test.plot(kind='kde', figsize=(15, 15), label='residual_test')
    # plt.title('residual_test')
    #
    # if directory is not None:
    #     path_pdf = f'{directory}/pdf_{file_name}.png'
    #     plt.savefig(path_pdf)

    fig.show()

    file_names = None
    pathes = None

    if directory is not None:

        file_name_train = f'res_train_{file_name}'
        file_name_test = f'res_test_{file_name}'
        file_names = [f'pdf_{file_name}', file_name_train, file_name_test]

        path_train = tsplot(error_train, title=f'Residual train {file_name}', path=f'{directory}/{file_name_train}')
        path_test = tsplot(error_test, title=f'Residual test {file_name}', path=f'{directory}/{file_name_test}')

        pathes = [path_pdf, path_train, path_test]
    else:
        tsplot(error_train, title='Residual train')
        tsplot(error_test, title='Residual test')

    return file_names, pathes

def inv_boxcox_prediction_and_realvalues(y, prediction_train, prediction_test, lmbd):
    inv_pred_train = inv_boxcox(prediction_train, lmbd)
    inv_pred_test = inv_boxcox(prediction_test, lmbd)
    inv_y = inv_boxcox(y, lmbd)

    inv_error_test = inv_y[-len(inv_pred_train):] - inv_pred_test
    inv_error_train = inv_y[:len(inv_pred_train)] - inv_pred_train

    return inv_y, inv_pred_train, inv_pred_test, inv_error_train, inv_error_train

def get_describin_from_residuals(error_train, error_test, columns_name=['train_ressiduals', 'test_ressiduals']):
    error_train_describe = error_train.describe()
    error_test_describe = error_test.describe()

    error_describing = pd.DataFrame({columns_name[0] : error_train_describe.values, columns_name[1]: error_test_describe}, index=error_train_describe.index)
    return error_describing

def get_error(y, prediction_train, prediction_test):
#     inv_pred_train = inv_boxcox(prediction_train, lmbd)
#     inv_pred_test = inv_boxcox(prediction_test, lmbd)
#     inv_y = inv_boxcox(y, lmbd)

    error_test = y[prediction_test.index] - prediction_test
    error_train = y[prediction_train.index] - prediction_train

    return error_train, error_test

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h