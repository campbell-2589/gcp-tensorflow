#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import pandas as pd
import os
import shutil


print(tf.__version__)

# Make sure the google cloud SDK is set up before running this
os.system('gsutil cp gs://cloud-training-demos/taxifare/small/*.csv .')
os.system('ls -l *.csv')

# We're using a subset of the ubiquitous NYC taxi data
df_train = pd.read_csv(filepath_or_buffer="./taxi-train.csv")
df_valid = pd.read_csv(filepath_or_buffer="./taxi-valid.csv")
df_test = pd.read_csv(filepath_or_buffer="./taxi-test.csv")

CSV_COLUMN_NAMES = list(df_train)
print(CSV_COLUMN_NAMES)

FEATURE_NAMES = CSV_COLUMN_NAMES[1:]  # all but first column
LABEL_NAME = CSV_COLUMN_NAMES[0]  # first column

feature_columns = [tf.feature_column.numeric_column(key=k) for k in FEATURE_NAMES]


def train_input_fn(df, batch_size=128):
    # Convert dataframe into (features,label) format for Estimator API
    dataset = tf.data.Dataset.from_tensor_slices(tensors=(dict(df[FEATURE_NAMES]), df[LABEL_NAME]))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size=batch_size)

    return dataset


def eval_input_fn(df, batch_size=128):
    # Convert dataframe into  (features,label) format for Estimator API
    dataset = tf.data.Dataset.from_tensor_slices(tensors=(dict(df[FEATURE_NAMES]), df[LABEL_NAME]))

    # Batch the examples.
    dataset = dataset.batch(batch_size=batch_size)

    return dataset


def predict_input_fn(df, batch_size=128):
    # Convert dataframe into  (features) format for Estimator API
    dataset = tf.data.Dataset.from_tensor_slices(tensors=dict(df[FEATURE_NAMES]))  # no label

    # Batch the examples.
    dataset = dataset.batch(batch_size=batch_size)

    return dataset


# Choices for LM estimators in TF
# LinearClassifier/Regressor
# BoostedTreesClassifier/Regressor
# DNNClassifier/Regressor
# DNNLinearCombinedClassifier/Regressor

OUTDIR = "taxi_trained"

model = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    model_dir=OUTDIR,
    config=tf.estimator.RunConfig(tf_random_seed=1)  # for reproducibility
)

tf.logging.set_verbosity(tf.logging.INFO)  # so loss is printed during training
shutil.rmtree(path=OUTDIR, ignore_errors=True)  # start fresh each time
model.train(input_fn=lambda: train_input_fn(df=df_train), steps=500)


def print_rmse(model, df):
    metrics = model.evaluate(input_fn=lambda: eval_input_fn(df))
    print("RMSE on dataset = {}".format(metrics["average_loss"] ** .5))


print_rmse(model=model, df=df_valid)

predictions = model.predict(input_fn=lambda: predict_input_fn(df=df_test[:10]))
for items in predictions:
    print(items)

# Model is pretty bad-  it predicts similar amounts for every trip.

tf.logging.set_verbosity(tf.logging.INFO)

shutil.rmtree(path=OUTDIR, ignore_errors=True)
model = tf.estimator.DNNRegressor(
    hidden_units=[10, 10],  # specify neural architecture
    feature_columns=feature_columns,
    model_dir=OUTDIR,
    config=tf.estimator.RunConfig(tf_random_seed=1)
)
model.train(
    input_fn=lambda: train_input_fn(df=df_train),
    steps=500)

print_rmse(model=model, df=df_valid)

# Performance is only slightly better at 9.26, and still far worse than our rules based model.  This illustrates an important tenant of machine learning: A more complex model can't outrun bad data.
#
# Currently since we're not doing any feature engineering our input data has very little signal to learn from, so using a DNN doesn't help much.
