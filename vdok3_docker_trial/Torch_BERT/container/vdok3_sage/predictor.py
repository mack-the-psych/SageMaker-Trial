# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

# Makoto.Sano@Mack-the-Psych.com
from torch_bert_classify import tmv_torch_bert_classify

dependent_var = r'Definition-Score'
number_class = 3
# csv_dump = False
# batch_size = 100
# epochs = 1
task_word = r'Definition'
# key_word = r'TORCH_BERT-Def-PRE-All'
answer_ex_clm = 'Definition'
lang = [0, 1]

import os
import json
import pickle
# import StringIO
# from io import StringIO
import io as StringIO
import sys
import signal
import traceback

import flask

import numpy as np
import pandas as pd

import cloudpickle
import torch 
from torch import nn

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

input_path = prefix + 'input/data'
channel_name='training'
tmp_csv_name = 'TORCH_RESPONSE_ANSWER_EX_FILE.CSV'
batch_size = 32

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            # Makoto.Sano@Mack-the-Psych.com
            bertd = tmv_torch_bert_classify('/opt/program/vdok3/data/', '/opt/program/vdok3/train/pytorch_advanced/nlp_sentiment_bert/')
            # bertd.restore_model(key_word)            
            bertd.modeling_data_file_name = bertd.data_dir + tmp_csv_name
            bertd.batch_size = batch_size
            with open(os.path.join(model_path, 'vdok3_bert.pkl'), 'rb') as f:
                bertd.net_trained = cloudpickle.load(f)
            bertd.criterion = nn.CrossEntropyLoss()
            
            cls.model = bertd
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()        
        
        # Makoto.Sano@Mack-the-Psych.com        
        answer_clm = 'Definition-Answer'
        answer_ex_clm = task_word

        clf.dependent_var = dependent_var
        clf.answer_ex_clm = answer_ex_clm
        # self.df_response_answer_ex = self.df_response_answer_ex.set_index(r'Student_Question_Index')
        input = input.set_index(r'Student_Question_Index')
          
        clf.ans_clm = task_word + r'-Answer'
        clf.ans_and_ex_clm = task_word + r'-Answer-and-Example'
        
        input[clf.ans_and_ex_clm] = input[clf.answer_ex_clm] + ' ' + input[clf.ans_clm]

        # to move LABEL and TXT columns to the end
        columns = list(input.columns)
        columns.remove(clf.dependent_var)
        columns.remove(clf.ans_and_ex_clm)
        columns.append(clf.dependent_var)
        columns.append(clf.ans_and_ex_clm)
        clf.df_ac_modeling_values = input.reindex(columns=columns)        
        clf.df_response_answer_ex = input
        clf.perform_prediction(clf.df_ac_modeling_values, number_class)
              
        return clf.df_ac_classified
    
# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = StringIO.StringIO(data)
        # Makoto.Sano@Mack-the-Psych.com
        # data = pd.read_csv(s, header=None)
        data = pd.read_csv(s)
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    # Makoto.Sano@Mack-the-Psych.com
    # print('Invoked with {} records'.format(data.shape[0]))
    print('Invoked with {} records'.format(len(data)))
    
    # Do the prediction
    predictions = ScoringService.predict(data)

    # Convert from numpy back to CSV
    out = StringIO.StringIO()

    # Makoto.Sano@Mack-the-Psych.com
    # pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
    predictions.to_csv(out, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')
