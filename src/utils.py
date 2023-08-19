import json
import os,sys
from src.exception import CustomException
import tensorflow as tf
import re, string


def custom_standardization(input_data):
  try:
    std_text = tf.strings.lower(input_data)
    std_text = tf.strings.regex_replace(std_text, r"https:\/\/.*[\r\n]*", '')
    std_text = tf.strings.regex_replace(std_text, r"www\.\w*\.\w\w\w", '')
    std_text = tf.strings.regex_replace(std_text, r"<[\w]*[\s]*/>", '')
    std_text = tf.strings.regex_replace(std_text, '[%s]' % re.escape(string.punctuation), '')
    std_text = tf.strings.regex_replace(std_text, '\s{2}', '')
    std_text = tf.strings.strip(std_text)
    return std_text
  except Exception as e:
     raise CustomException(e,sys)

def save_json_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'w') as f:
            json.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)
