#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import pandas as pd
import os
import shutil

print(tf.__version__)

CSV_COLUMN_NAMES = ["fare_amount", "dayofweek", "hourofday", "pickuplon", "pickuplat", "dropofflon", "dropofflat"]
CSV_DEFAULTS = [[0.0], [1], [0], [-74.0], [40.0], [-74.0], [40.7]]


def read_dataset(csv_path):
    def _parse_row(row):
        # Decode the CSV row into list of TF tensors
        fields = tf.decode_csv(records=row, record_defaults=CSV_DEFAULTS)

        # Pack the result into a dictionary
        features = dict(zip(CSV_COLUMN_NAMES, fields))

        # NEW: Add engineered features
        features = add_engineered_features(features)

        # Separate the label from the features
        label = features.pop("fare_amount")  # remove label from features and store

        return features, label

    # Create a dataset containing the text lines.
    dataset = tf.data.Dataset.list_files(file_pattern=csv_path)  # (i.e. data_file_*.csv)
    dataset = dataset.flat_map(map_func=lambda filename: tf.data.TextLineDataset(filenames=filename).skip(count=1))

    # Parse each CSV row into correct (features,label) format for Estimator API
    dataset = dataset.map(map_func=_parse_row)

    return dataset


def train_input_fn(csv_path, batch_size=128):
    # 1. Convert CSV into tf.data.Dataset with (features,label) format
    dataset = read_dataset(csv_path)

    # 2. Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size=batch_size)

    return dataset


def eval_input_fn(csv_path, batch_size=128):
    # 1. Convert CSV into tf.data.Dataset with (features,label) format
    dataset = read_dataset(csv_path)

    # 2.Batch the examples.
    dataset = dataset.batch(batch_size=batch_size)

    return dataset


# 1. One hot encode dayofweek and hourofday
fc_dayofweek = tf.feature_column.categorical_column_with_identity(key = "dayofweek", num_buckets = 7)
fc_hourofday = tf.feature_column.categorical_column_with_identity(key = "hourofday", num_buckets = 24)

# 2. Bucketize latitudes and longitudes
NBUCKETS = 16
latbuckets = np.linspace(start = 38.0, stop = 42.0, num = NBUCKETS).tolist()
lonbuckets = np.linspace(start = -76.0, stop = -72.0, num = NBUCKETS).tolist()
fc_bucketized_plat = tf.feature_column.bucketized_column(source_column = tf.feature_column.numeric_column(key = "pickuplon"), boundaries = lonbuckets)
fc_bucketized_plon = tf.feature_column.bucketized_column(source_column = tf.feature_column.numeric_column(key = "pickuplat"), boundaries = latbuckets)
fc_bucketized_dlat = tf.feature_column.bucketized_column(source_column = tf.feature_column.numeric_column(key = "dropofflon"), boundaries = lonbuckets)
fc_bucketized_dlon = tf.feature_column.bucketized_column(source_column = tf.feature_column.numeric_column(key = "dropofflat"), boundaries = latbuckets)

# 3. Cross features to get combination of day and hour
fc_crossed_day_hr = tf.feature_column.crossed_column(keys = [fc_dayofweek, fc_hourofday], hash_bucket_size = 24 * 7)


def add_engineered_features(features):
    features["dayofweek"] = features["dayofweek"] - 1  # subtract one since our days of week are 1-7 instead of 0-6

    features["latdiff"] = features["pickuplat"] - features["dropofflat"]  # East/West
    features["londiff"] = features["pickuplon"] - features["dropofflon"]  # North/South
    features["euclidean_dist"] = tf.sqrt(x=features["latdiff"] ** 2 + features["londiff"] ** 2)

    return features

#
# Gather list of feature columns
# Ultimately our estimator expects a list of feature columns, so let's gather all our engineered features into a single list.
#
# We cannot pass categorical or crossed feature columns directly into a DNN, Tensorflow will give us an error. We must first wrap them using either indicator_column() or embedding_column(). The former will pass through the one-hot encoded representation as is, the latter will embed the feature into a dense representation of specified dimensionality (the 4th root of the number of categories is a good starting point for number of dimensions). An embedding column stores categorical data in a lower-dimensional vector than an indicator column.
# A general rule of thumb about the number of embedding dimensions: embedding_dimensions =  number_of_categories**0.25