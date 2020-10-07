from __future__ import absolute_import, division, print_function, unicode_literals

import tkinter as tk
from tkinter import filedialog as fd
import tensorflow as tf
import pandas as pd
import time

t1 = time.time()
clogging = ['normal', 'clog']

def getFile(action):
    print(f'Getting {action} file selected')
    root = tk.Tk()
    filename = fd.askopenfile(title=f'Select your {action} DATASETS!')
    root.destroy()
    return filename

# Use keras to grab datasets and read them into pandas dataframe
dtrain = pd.read_csv(getFile('Training'))
dtest = pd.read_csv(getFile('Evaluation'))
dpred = pd.read_csv(getFile('Prediction'))

dtrain_y = dtrain.pop('clog')
dtest_y = dtest.pop('clog')
dpred_y = dpred.pop('clog')


# Make input functions
def input_fn(features, labels, training=True, batch_size=256):
    # make datasets
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # shuffle and repeat the data feeding
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)


# Define Feature Columns/Keys
feat_col = []

for key in dtrain.keys():
    feat_col.append(tf.feature_column.numeric_column(key=key))

# CREATING THE MODEL
classifier = tf.estimator.DNNClassifier(
    feature_columns=feat_col,  # Specify the feature of data
    hidden_units=[30, 10],  # specify the layer shape (2 hidden layers, 1st layer => 30 nodes, 2nd layer => 10 nodes)
    n_classes=2)  # how many class to predict

# Train The Model
classifier.train(
    input_fn=lambda: input_fn(dtrain, dtrain_y, True),
    # input function can be defined to make a function by using lambda
    steps=5000)  # similar with epochs, how many times data will be read (studied)

# Checking the model specification
result_eval = classifier.evaluate(input_fn=lambda: input_fn(dtest, dtest_y, False))
print(result_eval)


# Define prediction input function
def pred_input_fn(features, batch_size=256):  # converting into datasets
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

# ==================================================

def predict_dataset(dataset):
    # setup the predict dictionary datasets
    feat_col = []
    predicted_val = []
    for f in dataset.columns:
        feat_col.append(f)

    for i in range(len(dataset)):
        print(f'==========================\nData Remaining: {len(dataset) - i}\n==========================')
        fpred = {}
        for ii in enumerate(feat_col):
            fpred[ii[1]] = [dataset.loc[i][ii[0]]]
        # do predicting using trained DNN model
        predictions = classifier.predict(input_fn=lambda: pred_input_fn(fpred))
        for pred_dict in predictions:
            class_id = pred_dict['class_ids'][0]
            predicted_val.append(class_id)

    cols = []
    for i in dtrain.columns:
        cols.append(i)

    datass = []
    for i in enumerate(predicted_val):
        datass.append([])
        datass[i[0]].append(i[1])
        for ii in cols:
            datass[i[0]].append(dpred.loc[i[0]][ii])

    cols.insert(0, 'clog')
    pred_df = pd.DataFrame(datass, columns=cols)
    return pred_df


predicted_DF = predict_dataset(dpred)

print(f'Elapsed Time: {round((time.time() - t1), 3)} s')
predicted_DF.to_csv('predicted/dnn_results_normal.csv',index=False)