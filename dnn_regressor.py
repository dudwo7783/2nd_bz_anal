# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import itertools
import statsmodels.api as sm
from pylab import rcParams
import statsmodels.stats.stattools as smtools
import matplotlib.backends.backend_pdf
#import seaborn as sns
from os.path import join, os
from scipy import stats
from sklearn.svm import SVR, SVC
from sklearn.feature_selection import RFECV, SelectFromModel, SelectKBest, chi2, SelectPercentile
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
import time as tm
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import _pickle as cPickle
from scipy.stats import pearsonr

# RTDB 경로
RTDB_path = r'C:\Users\Administrator\Desktop\KYJ\coke\regen_feature_tune_data\RTDB.csv'
data_dir = r'C:\Users\Administrator\Desktop\KYJ\coke\2nd_bz_anal_data'

def DrawTimePlot(x,y, title):
    fig = plt.figure()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    
    # Add axis labels
    plt.xlabel('time')
    plt.ylabel('2nd.bz.max.temp')
    plt.title(title)

    plt.plot(x,y)
    #fig.savefig(join(data_dir,title + '.pdf'))
    
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def getDataFrame(path):
    df = pd.read_csv(path, index_col=0)
    return df

def strtodate(date_arr):
    x = [dt.datetime.strptime(d,'%Y-%m-%d %H:%M') for d in date_arr]
    return x

def RemoveLowerOut(df,column_name):
    deout_df = df[(stats.zscore(df[column_name]) > -3)]
    return deout_df

def main(_):
    RTDB = getDataFrame(RTDB_path)
    RTDB = RTDB.drop(['Catalyst_V3_front_cl Content', 'Catalyst_V3_back_cl Content'],1)

    # What is the more userful?
    
    RTDB = RemoveLowerOut(RTDB,'max_2nd.Burning.Zone')
    #DrawTimePlot(date,max_2nd_bz, 'Remove outlier max temp plot')
    
    idx_RTDB = RTDB.reset_index() # Remove NaN row

    
    x_RTDB = idx_RTDB.drop(['max_2nd.Burning.Zone', 'max_2nd.1dan', 'max_2nd.2dan', 
                             'max_2nd.3dan', 'max_2nd.4dan', 'max_2nd.5dan', 'index'],1)
    idx_RTDB = idx_RTDB.drop(['max_2nd.1dan', 'max_2nd.2dan', 
                             'max_2nd.3dan', 'max_2nd.4dan', 'max_2nd.5dan', 'index'],1)
    
    x_RTDB = x_RTDB[:len(x_RTDB)-120]
    idx_RTDB = idx_RTDB[:len(idx_RTDB)-120]
    
    train_x = x_RTDB.loc[1:(len(idx_RTDB)*4/5)]
    train_y = idx_RTDB.loc[1:(len(idx_RTDB)*4/5),['max_2nd.Burning.Zone']]
    
    test_x = x_RTDB.loc[(len(idx_RTDB)*4/5):]
    test_y = idx_RTDB.loc[(len(idx_RTDB)*4/5):, ['max_2nd.Burning.Zone']]

    dnn_train_x = idx_RTDB.loc[:(len(idx_RTDB)*4/5)]
    dnn_train_x.columns = [col.replace(' ', '_') for col in dnn_train_x.columns]
    dnn_train_y = idx_RTDB.loc[(len(idx_RTDB)*4/5):]
    dnn_train_y.columns = [col.replace(' ', '_') for col in dnn_train_x.columns]

    scaler = preprocessing.StandardScaler().fit(train_x)
    scale_train = scaler.transform(train_x) 
    scale_test = scaler.transform(test_x)
    
    scaled_train_df = pd.DataFrame(
        scale_train, index=train_x.index, columns=train_x.columns)
    
    scaled_test_df = pd.DataFrame(
        scale_test, index=test_x.index, columns=test_x.columns)
    
    real = test_y['max_2nd.Burning.Zone']
    
    feature_cols = [tf.contrib.layers.real_valued_column(k) for k in scaled_train_df.columns.values]
    
    regressor = tf.contrib.learn.DNNRegressor(
    feature_columns=feature_cols, hidden_units=[1024, 512, 256], optimizer= tf.train.AdamOptimizer(),
    model_dir = './reg_model'
    )
    
    regressor.fit(x=scaled_train_df, y=train_y, steps=5000)
    
    # Score accuracy
    ev = regressor.evaluate(x=scaled_test_df, y=real, steps=1)
    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))
    
    # Print out predictions    
    y = regressor.predict(scaled_test_df)
    predictions = list(itertools.islice(y, len(scaled_test_df)))
    #print("Predictions: {}".format(str(predictions)))
    #print("\n\n")
    #print(predictions)
    
    print("\n\nRMSE:")
    print(np.sqrt(((real - predictions) **2).mean()))
    
    x = np.arange(len(real))
    plt.xticks(x, real, rotation='vertical')
    plt.yticks(  fontsize = 20)
    #plt.plot(x,real_value[1], color = 'r')

    plt.gca().set_color_cycle(['red', 'green'])
    plt.plot(x,real)
    plt.plot(x,predictions)
    legend = plt.legend(['real value', 'predicted value'])
    legend.get_title().set_fontsize('10') #legend 'Title' fontsize
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='30') #legend 'list' fontsize
    plt.show()
    plt.plot(real - predictions)
    plt.show()
    
    rcParams['figure.figsize'] = 10, 10
    sm.qqplot(real - predictions, line='r')
    plt.show()

    
    print("\n\n durbin watson test :")
    print(smtools.durbin_watson(real -predictions, axis=0))

if __name__ == "__main__":
    tf.app.run()