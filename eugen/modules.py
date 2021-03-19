import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import r2_score
# from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import explained_variance_score


def barplot(data, title):
#     fig = plt.figure(figsize=(18,6))
    bar_plot = sns.barplot(x=data['feature'], y=data['value'])
    for item in bar_plot.get_xticklabels():
        item.set_rotation(90)
    plt.title(title)
    plt.show()

def get_score_for_model(models, X_train, y_train, n_splits=10, scoring='roc_auc', print_res=True):
    def append_res_to_boxplot():
        i = 0
        df = pd.DataFrame()
        while i < len(results[0]):
            line = [[num[i], ml] for num, ml in zip(results, names)]
            #             for num, ml in zip(results, names):
            #                 line.append([num[i],ml])
            i = i + 1
            df = df.append(pd.DataFrame(line, columns=[scoring, 'ML']), ignore_index=True)
        return df

    seed = 13

    results = []
    means = []
    sdv = []
    names = []
    scoring = scoring

    for name, model in models:
        strat = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)

        cv_results = cross_val_score(model, X_train, y_train, cv=strat, scoring=scoring, n_jobs=-1)

        results.append(cv_results)
        names.append(name)
        means.append(cv_results.mean())
        sdv.append(cv_results.std())

        if print_res:
            print(f"{names[-1]}: {means[-1]} ({sdv[-1]})")

    box_plot = append_res_to_boxplot()

    df_means = pd.DataFrame({'ML': names, 'means': means, 'std': sdv})
    return box_plot, df_means


def define_metrics(model, X_train_, X_test_, y_train, y_test, name):
    pred_train_ = np.array(model.predict(X_train_))
    pred_test_ = np.array(model.predict(X_test_))
    y_train_ = np.array(y_train)
    y_test_ = np.array(y_test)

    metric_train = pd.DataFrame()
    metric_train['name'] = [name + '_train']
    metric_train['r2'] = [r2_score(y_train, pred_train_)]
    metric_train['sum_squared_resid'] = np.sum((y_train_ - pred_train_)**2)
    metric_train['MAPE'] = [np.mean(np.abs((y_train - pred_train_) / y_train)) * 100]
    metric_train['RMSE'] = [np.sum((y_train - pred_train_)**2) / len(y_train_)]
    metric_train['durbin_watson'] = [durbin_watson(y_train - pred_train_)]
    metric_train['theil_index'] = [np.sqrt((1/len(pred_train_))*np.sum((y_train_-pred_train_)**2))
                                      / (np.sqrt((1/len(y_train_))*np.sum(y_train_**2)) + np.sqrt((1/len(pred_train_))*np.sum(pred_train_**2)))]
    
    metric_train['ex_var'] = [explained_variance_score(y_train, pred_train_)]

        
    metric_test = pd.DataFrame()
    metric_test['name'] = [name + '_test']
    metric_test['r2'] = [r2_score(y_test, pred_test_)]
    metric_train['sum_squared_resid'] = np.sum((y_test_ - pred_test_)**2)
    metric_test['MAPE'] = [np.mean(np.abs((y_test - pred_test_) / y_test)) * 100]
    metric_train['RMSE'] = [np.sum((y_test - pred_test_) ** 2) / len(pred_test_)]
    metric_test['durbin_watson'] = [durbin_watson(y_test - pred_test_)]
    metric_train['theil_index'] = [np.sqrt((1/len(pred_test_))*np.sum((y_test_-pred_test_)**2))
                                      / (np.sqrt((1/len(y_test_))*np.sum(y_test_**2)) + np.sqrt((1/len(pred_test_))*np.sum(pred_test_**2)))]
    
    metric_test['ex_var'] = [explained_variance_score(y_test, pred_test_)]
    

    return metric_train.append(metric_test)


if __name__ == '__main__':
    pass
