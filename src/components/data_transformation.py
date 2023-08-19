import os,sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

from src.logger import logging
from src.exception import CustomException
from src.utils import custom_standardization


class DataTransformation():

    def initiate_data_transformation(self,trainset_path,testset_path):
        try:
            train_df = pd.read_csv(trainset_path)
            test_df = pd.read_csv(testset_path)
            target_column_name = 'sentiment'

            logging.info("Initiating Data Transformation")

            train_df[target_column_name] = train_df[target_column_name].apply(lambda row : 1 if row == 'positive' else 0)
            test_df[target_column_name] = test_df[target_column_name].apply(lambda row : 1 if row == 'positive' else 0)

            logging.info("Converting dataframe to array")
            train_df_features = train_df['review'].to_numpy()
            train_df_target = train_df['sentiment'].to_numpy()

            test_df_features = test_df['review'].to_numpy()
            test_df_target = test_df['sentiment'].to_numpy()

            logging.info("Splitting the dataset into train,valid and test set")
            features_train, features_test, labels_train, labels_test = train_test_split(train_df_features, train_df_target, test_size=0.4, random_state = 0)
            features_valid, features_test, labels_valid, labels_test = train_test_split(test_df_features, test_df_target, test_size=0.5, random_state=0)

            logging.info("Converting the arrays to tensors.")
            features_train = tf.convert_to_tensor(features_train)
            labels_train = tf.convert_to_tensor(labels_train)

            features_valid = tf.convert_to_tensor(features_valid)
            labels_valid = tf.convert_to_tensor(labels_valid)

            features_test = tf.convert_to_tensor(features_test)
            labels_test = tf.convert_to_tensor(labels_test)

            logging.info("Creating Tensor Slices and batches of 64.")
            train_ds = tf.data.Dataset.from_tensor_slices((features_train, labels_train))
            valid_ds = tf.data.Dataset.from_tensor_slices((features_valid, labels_valid))
            test_ds = tf.data.Dataset.from_tensor_slices((features_test, labels_test))

            train_ds = train_ds.batch(batch_size=64)
            valid_ds = valid_ds.batch(batch_size=64)
            test_ds = test_ds.batch(batch_size=64)

            logging.info("Auto-tuning the dataset.")
            train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            valid_ds = valid_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

            logging.info("Generating the vectorizer layer.")
            vectorizer_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=1000,
                                                                                standardize=custom_standardization)
            vectorizer_layer.adapt(train_ds.map(lambda text, label: text))

            logging.info("Data Transformation Complete.")
            return(
                train_ds,valid_ds,test_ds,vectorizer_layer
            )
        except Exception as e:
            raise CustomException(e,sys)
