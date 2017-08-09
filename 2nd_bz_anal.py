import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import matplotlib.backends.backend_pdf
#import seaborn as sns
from os.path import join, os
from scipy import stats
from sklearn.svm import SVR, SVC
from sklearn.feature_selection import RFECV, SelectFromModel, RFE
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
import time as tm
import _pickle as cPickle

# RTDB 경로
RTDB_path = r'C:\Users\Administrator\Desktop\KYJ\coke\regen_feature_tune_data\RTDB.csv'
data_dir = r'C:\Users\Administrator\Desktop\KYJ\coke\2nd_bz_anal_data'

def getDataFrame(path):
    df = pd.read_csv(path, index_col=0)
    return df

def strtodate(date_arr):
    x = [dt.datetime.strptime(d,'%Y-%m-%d %H:%M').date() for d in date_arr]
    return x

def RemoveLowerOut(df,column_name):
    deout_df = df[(stats.zscore(df[column_name]) > -3)]
    return deout_df


def DrawTimePlot(x, y, title):
    fig = plt.figure()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    # Add axis labels
    plt.xlabel('time')
    plt.ylabel('2nd.bz.max.temp')
    plt.title(title)

    plt.plot(x, y)
    plt.show()
    fig.savefig(join(data_dir, title + '.pdf'))


def DrawTempPlt(target, df):
    dan_path = join(data_dir, "max_2nd.Burning.Zone")

    if not os.path.exists(dan_path):
        os.makedirs(dan_path)

    full_fig, full_ax = plt.subplots()

    y_min = target.min()  # 2nd bz temp min
    y_max = target.max()  # 2nd bz temp max

    full_fig, full_ax = plt.subplots()
    y = target

    for tag in df.columns.values:
        part_fig, part_ax = plt.subplots(subplot_kw=dict(ylim=(y_min, y_max)))
        x = df[tag]
        # full_ax.scatter(x, y, c=color, marker='o', s=5)
        part_ax.scatter(x, y, s=5)
        part_ax.set_title("%s" % (tag + " vs max_2nd.Burning.Zone"))
        part_fig.savefig(join(dan_path, (tag + " vs max_2nd.Burning.Zone.png")))

    plt.clf()

    # full_ax.set_title("Full")
    # full_fig.savefig("full_png")


def DrawDanTempPlt(df):
    columns = df.columns.values
    bz_dan_list = [col for col in columns if 'dan' in col]

    for dan in bz_dan_list:
        dan_path = join(data_dir, dan)

        if not os.path.exists(dan_path):
            os.makedirs(dan_path)

        y = df[dan]
        y_min = y.min()  # 2nd bz temp min
        y_max = y.max()  # 2nd bz temp max

        full_fig, full_ax = plt.subplots()

        for tag in df.columns.values:
            part_fig, part_ax = plt.subplots(subplot_kw=dict(ylim=(y_min, y_max)))
            x = df[tag]
            part_ax.scatter(x, y, s=5)
            part_ax.set_title("%s" % (tag + " vs " + dan))
            part_fig.savefig(join(dan_path, (tag + " vs " + dan + ".png")))

        plt.clf()


def feature_selection(x, y_, method):
    if "RFE" in method:
        estimator = SVR(cache_size=7000)
        a = estimator.fit(x, y_)
        # selector = SelectFromModel(estimator)
        # selector = selector.fit(x, y_)
        # print(selector.support_)
        # print(selector.ranking_)

        return a

    elif "DNN" in method:
        FEATURES = x.columns.values
        LABEL = "max_2nd.Burning.Zone"

        def input_fn(data_set):
            feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
            labels = tf.constant(data_set[LABEL].values)
            return feature_cols, labels

        # Feature cols
        feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

        # Build 2 layer fully connected DNN with 10, 10 units respectively.
        regressor = tf.contrib.learn.DNNRegressor(
            feature_columns=feature_cols, hidden_units=[128, 64, 32])
        # optimizer = tf.train.AdamOptimizer

        # Fit
        regressor.fit(input_fn=lambda: input_fn(x), steps=10000)

        # Score accuracy
        ev = regressor.evaluate(input_fn=lambda: input_fn(y_), steps=1)
        loss_score = ev["loss"]
        print("Loss: {0:f}".format(loss_score))

        # Print out predictions
        # y = regressor.predict(input_fn=lambda: input_fn(test_set))
        # predictions = list(itertools.islice(y, len(test_data)))
        # print("Predictions: {}".format(str(predictions)))
        # print("\n\n")
        # print(predictions)

    elif "gridcv" in method:
        # Create the RFE object and compute a cross-validated score.
        svr = SVR(kernel="linear")
        rfecv = RFECV(estimator=svr, step=1, cv=StratifiedKFold(2), scoring='accuracy', n_jobs=-1)

        parameters = {'C': [0.001, 0.01, 0.1, 1, 10], 'n_jobs': [4]}
        svr = SVR()
        clf = GridSearchCV(svr, parameters)

        selector = Pipeline([
            ('feature_selection', rfecv),
            ('regression', clf)
        ])
        selector.fit(x, y_)

        return selector

    else:
        # Create the RFE object and compute a cross-validated score.
        svc = SVC(kernel="linear")
        rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring='accuracy', n_jobs=-1)

        parameters = {'C': [0.001, 0.01, 0.1, 1, 10]}
        svr = SVR()
        clf = GridSearchCV(svr, parameters)

        selector = Pipeline([
            ('feature_selection', rfecv),
            ('regression', clf)
        ])
        selector.fit(x, y_)

        return selector


if __name__ == "__main__":
    RTDB = getDataFrame(RTDB_path)
    RTDB = RTDB.drop(['Catalyst_V3_front_cl Content', 'Catalyst_V3_back_cl Content'], 1)

    # What is the more userful?

    date = strtodate(RTDB.index)
    max_2nd_bz = RTDB['max_2nd.Burning.Zone']
    # max_2nd_bz = np.array(RTDB['max_2nd.Burning.Zone'])

    RTDB = RemoveLowerOut(RTDB, 'max_2nd.Burning.Zone')

    date = strtodate(RTDB.index)
    max_2nd_bz = RTDB['max_2nd.Burning.Zone']

    # DrawTimePlot(date,max_2nd_bz, 'Remove outlier max temp plot')

    idx_RTDB = RTDB.reset_index()  # Remove NaN row

    x_RTDB = idx_RTDB.drop(['max_2nd.Burning.Zone', 'max_2nd.1dan', 'max_2nd.2dan',
                            'max_2nd.3dan', 'max_2nd.4dan', 'max_2nd.5dan', 'index'], 1)
    idx_RTDB = idx_RTDB.drop(['max_2nd.1dan', 'max_2nd.2dan',
                              'max_2nd.3dan', 'max_2nd.4dan', 'max_2nd.5dan', 'index'], 1)

    x_RTDB = x_RTDB[:len(x_RTDB) - 120]
    idx_RTDB = idx_RTDB[:len(idx_RTDB) - 120]

    train_x = x_RTDB.loc[1:(len(idx_RTDB) * 4 / 5)]
    train_y = idx_RTDB.loc[1:(len(idx_RTDB) * 4 / 5), ['max_2nd.Burning.Zone']]

    test_x = x_RTDB.loc[(len(idx_RTDB) * 4 / 5):]
    test_y = idx_RTDB.loc[(len(idx_RTDB) * 4 / 5):, ['max_2nd.Burning.Zone']]

    short_x = x_RTDB.loc[1:(len(idx_RTDB) * 4 / 50)]
    short_y = idx_RTDB.loc[1:(len(idx_RTDB) * 4 / 50), ['max_2nd.Burning.Zone']]

    short_x_t = x_RTDB.loc[(len(idx_RTDB) * 4 / 50):(len(idx_RTDB) * 8 / 50)]
    short_y_t = idx_RTDB.loc[(len(idx_RTDB) * 4 / 50):(len(idx_RTDB) * 8 / 50), ['max_2nd.Burning.Zone']]

    dnn_train_x = idx_RTDB.loc[:(len(idx_RTDB) * 4 / 5)]
    dnn_train_x.columns = [col.replace(' ', '_') for col in dnn_train_x.columns]
    dnn_train_y = idx_RTDB.loc[(len(idx_RTDB) * 4 / 5):]
    dnn_train_y.columns = [col.replace(' ', '_') for col in dnn_train_x.columns]

    train_y_label = pd.DataFrame(train_y['max_2nd.Burning.Zone'].apply(lambda x: 1 if x >= 550 else 0))
    test_y_label = pd.DataFrame(test_y['max_2nd.Burning.Zone'].apply(lambda x: 1 if x >= 550 else 0))
    short_y_label = pd.DataFrame(short_y['max_2nd.Burning.Zone'].apply(lambda x: 1 if x >= 550 else 0))

    start_time = tm.time()
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear", verbose=True)
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
                  scoring='accuracy', verbose=10, n_jobs=2)
    rfecv.fit(short_x, short_y_label)
    print("rfecv-svm classification termination %s seconds ---" % (tm.time() - start_time))

    with open('rfecv-svm.pkl', 'wb') as fid:
        cPickle.dump(rfecv, fid)