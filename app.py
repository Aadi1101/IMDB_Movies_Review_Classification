from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import re,string,sys
from src.exception import CustomException

app = Flask('__name__')

@app.route('/')
def read_main():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def generate_output():
    json_data = False
    input_data = request.args.get('data')
    if input_data is None:
        input_data = request.get_json()
        json_data = True
    sentiment = process_and_predict(input_data=input_data, json_data=json_data)
    if sentiment<0.5:
        sentiment='positive'
    else:
        sentiment = 'negative'
    return {'sentiment': sentiment}

def process_and_predict(input_data:str, json_data):
    try:
        if json_data:
            output_data:str = input_data['data']
            output_data = output_data.replace('%20',' ')
        else:
            output_data:str = input_data
        print('\n')
        print(output_data)
        # Load the model
        model = tf.keras.models.load_model('./src/models/rnn_lstm_model/',custom_objects={'custom_standardization': custom_standardization})

        # Make predictions
        sentence = str(output_data)
        data = {'review':[sentence]}
        df = pd.DataFrame(data)
        features=df.to_numpy()
        features = tf.convert_to_tensor(features)

        sentiment = model.predict(features)

        return float(sentiment)  # Convert to float for JSON response
    except Exception as e:
        raise CustomException(e,sys)
def custom_standardization(input_data):
    std_text = tf.strings.lower(input_data)
    std_text = tf.strings.regex_replace(std_text, r"https:\/\/.*[\r\n]*", '')
    std_text = tf.strings.regex_replace(std_text, r"www\.\w*\.\w\w\w", '')
    std_text = tf.strings.regex_replace(std_text, r"<[\w]*[\s]*/>", '')
    std_text = tf.strings.regex_replace(std_text, '[%s]' % re.escape(string.punctuation), '')
    std_text = tf.strings.regex_replace(std_text, '\s{2}', '')
    std_text = tf.strings.strip(std_text)
    return std_text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)