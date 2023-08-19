import os,sys
import numpy as np
from dataclasses import dataclass

import tensorflow as tf

from src.logger import logging
from src.exception import CustomException
from src.utils import save_json_object

@dataclass
class ModelTrainerConfig():
    model_path = 'src/models/model'
    model_report_path = os.path.join('src/models','models_report.json')
class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_ds,valid_ds,test_ds,vectorizer_layer):
        try:
            model = tf.keras.Sequential([ vectorizer_layer,
                                        tf.keras.layers.Embedding(input_dim=1000, output_dim=64),
                                        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                                        tf.keras.layers.Dense(64, activation = 'relu'),
                                        tf.keras.layers.Dense(1, )
                                        ])

            model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          optimizer=tf.keras.optimizers.Adam(1e-4),
                          metrics=['accuracy']
                          )
            model.summary()
            history = model.fit(train_ds,epochs=10,validation_data=valid_ds)

            test_loss,test_acc = model.evaluate(test_ds)
            logging.info(f'loss value: {test_loss}')
            logging.info(f'Accuracy: {test_acc}')
            model.save('src/models/rnn_lstm_model')
            save_json_object(self.model_trainer_config.model_report_path,{"Recurrent Neural Network LSTM:":test_acc})
            test_predictions = model.predict(test_ds).flatten()
            logging.info("Model Training Completed.")
        except Exception as e:
            raise CustomException(e,sys)
