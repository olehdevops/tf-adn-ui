import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import explained_variance_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import train_test_split
from flask import Flask, request
import requests
import json

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predictions():

    data_from_api = json.loads(request.data.decode("utf-8"))

    if data_from_api == "test":
        df = pd.read_csv('dataset.csv').set_index('date')

        df = df.drop(['mintempm', 'maxtempm'], axis=1)

        # X will be a pandas dataframe of all columns except meantempm
        X = df[[col for col in df.columns if col != 'meantempm']]

        # y will be a pandas series of the meantempm
        y = df['meantempm']

        X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=23)
        X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=23)

        feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]

        regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                              hidden_units=[50, 50],
                                              model_dir='tf_wx_model')

        def wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=400):
            return tf.estimator.inputs.pandas_input_fn(x=X, y=y, num_epochs=num_epochs, shuffle=shuffle,
                                                       batch_size=batch_size)

        pred = regressor.predict(input_fn=wx_input_fn(X_test,
                                                      num_epochs=1,
                                                      shuffle=False))
        predictions = np.array([p['predictions'][0] for p in pred])

        pred_data = []
        for i in predictions:
            pred_data.append(float(i))

        test_data = []
        for i in y_test:
            test_data.append(float(i))

        demo_data = json.dumps({"pred_data": pred_data, "test_data": test_data})

        return demo_data


    else:

        records =[]
        city = ""
        for k, v in data_from_api.items():
            city = k
            records = v

        res_pred = {}

        features = ["date", "meantempm", "meandewptm", "meanpressurem", "maxhumidity", "minhumidity", "maxtempm",
                    "mintempm", "maxdewptm", "mindewptm", "maxpressurem", "minpressurem", "precipm"]


        df = pd.DataFrame(records, columns=features).set_index('date')
        tmp = df[['meantempm', 'meandewptm']].head(5)

        N = 1

        feature = 'meantempm'

        rows = tmp.shape[0]
        nth_prior_measurements = [None] * N + [tmp[feature][i - N] for i in range(N, rows)]

        col_name = "{}_{}".format(feature, N)
        tmp[col_name] = nth_prior_measurements

        def derive_nth_day_feature(df, feature, N):
            rows = df.shape[0]
            nth_prior_meassurements = [None] * N + [df[feature][i - N] for i in range(N, rows)]
            col_name = "{}_{}".format(feature, N)
            df[col_name] = nth_prior_meassurements

        for feature in features:
            if feature != 'date':
                for N in range(1, 4):
                    derive_nth_day_feature(df, feature, N)

        to_remove = [feature for feature in features if feature not in ['meantempm', 'mintempm', 'maxtempm']]

        to_keep = [col for col in df.columns if col not in to_remove]
        df = df[to_keep]
        df = df.apply(pd.to_numeric, errors='coerce')

        for precip_col in ['precipm_1', 'precipm_2', 'precipm_3']:
            missing_vals = pd.isnull(df[precip_col])
            df[precip_col][missing_vals] = 0

        df = df.dropna()
        df = df.drop(['mintempm', 'maxtempm', 'meantempm'], axis=1)

        X = df[[col for col in df.columns if col != 'meantempm']]

        feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]

        regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                              hidden_units=[50, 50],
                                              model_dir='tf_wx_model')

        def wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=400):
            return tf.estimator.inputs.pandas_input_fn(x=X, y=y, num_epochs=num_epochs, shuffle=shuffle,
                                                       batch_size=batch_size)

        pred = regressor.predict(input_fn=wx_input_fn(X,
                                                      num_epochs=1,
                                                      shuffle=False))
        predictions = np.array([p['predictions'][0] for p in pred])

        res_pred[city] = round(float(predictions[0]), 2)

        return json.dumps(res_pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
