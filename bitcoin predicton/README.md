# Bitcoin prediction for 1 day
Predict stock value of Bitcoin price. I used Yahoo API for gathering stock value and Google Trends API.

After gathering normalized values of searcing words with 85% of missing values. Used linear interpolation for filling missing values. Next step is to concatenated the Bitckoin time series price and other time series (amount of searching words “bircoin”, “BTC”, “bitcoin exhange”, “cryptocurrency” etc.)

Saved trained models, lags count, features importances, image with prediction on train and test in MLFlow. Too saved statisctics for residual like mean, std, autocorrelation, partial autocorrelation, QQ-plot and PDF for train and test set.

**Feature engeneering**: got from each time series lags, moving average, exponential moving average, moving standart deviation, exponential standart deviation. Prepared data like regression task. <br>
**Research**: Compared result using Box-Cox transformation and not. Worked with not stationar time series.<br>
**Libraries**: scikit-learn, Pandas, Numpy, Featuretools, SciPy, Matplotlib, Pandas-Profiling, yahoofinance, pytrends, statsmodels, MLFlow.<br>
**Models** : RandomForest, XGBoost, LinearRegression, Extra Trees, Decision Tree, Elastic Net.<br>


## Files:
- Classes for feaures engeenering. You can see .py file [here.](https://github.com/bateikoEd/dipl_program/blob/main/bitcoin%20predicton/feature_engeeniring.py)
- Class for combining gathered data from APIes. You can see .py file [here.](https://github.com/bateikoEd/dipl_program/blob/main/bitcoin%20predicton/pipeline_gatehering.py)
- Additional fucntions for creating images, metrics and saving in MLFlow. You can see .py file [here.](https://github.com/bateikoEd/dipl_program/blob/main/bitcoin%20predicton/functions.py)
- Notebook for full process of gatehering, preparing data and tuning models. You can see [here.](https://nbviewer.jupyter.org/github/bateikoEd/dipl_program/blob/main/bitcoin%20predicton/data_gathering_preparation.ipynb)
- In notebook we can see all time series before and after Box-Cox transforming. Too can see statistics for each time series and corelation matrix using pandas-profiling. You can see [here.](https://nbviewer.jupyter.org/github/bateikoEd/dipl_program/blob/main/bitcoin%20predicton/template_data.ipynb)


## Some insights
On this curves we can see all amount of google requests of word which were gathered from Google Trends.

<img src="../images/2021-05-07_22-00.png" data-canonical-src="../images/2021-05-07_22-00.png" width="900"/>
