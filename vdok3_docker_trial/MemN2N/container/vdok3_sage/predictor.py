# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function
# Makoto.Sano@Mack-the-Psych.com
from tmv_tf_memn_classify import tmv_tf_memn_classify

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

import pandas as pd

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

dependent_var = r'Definition-Score'
drop_vars = None

number_class = 3

# Makoto.Sano@Mack-the-Psych.com
class tmv_tf_memn_classify_in(tmv_tf_memn_classify):
    # Makoto.Sano@Mack-the-Psych.com
    def load_data(self, df_in, dependent_var, langs = None, task_word = 'Definition',
                  answer_ex_clm = 'Definition'):
        # Modified by mack.sano@gmail.com 3/22/2020
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        os.mkdir(LOG_DIR)

        self.dependent_var = dependent_var
        # Makoto.Sano@Mack-the-Psych.com
        self.df_response_answer_ex = df_in
        self.df_response_answer_ex = self.df_response_answer_ex.set_index(r'Student_Question_Index')

        if langs != None:
            lang_clm = task_word + r'-Language'
            self.df_response_answer_ex = \
                self.df_response_answer_ex[self.df_response_answer_ex[lang_clm].isin(langs)]
            
        ans_clm = task_word + r'-Answer'

        ans_tokens_all = self.get_tokens(ans_clm)
        ans_ex_tokens_all = self.get_tokens(answer_ex_clm)

        self.vocab = set()
        for x in ans_tokens_all + ans_ex_tokens_all:
            self.vocab |= set(x)
        self.vocab = sorted(self.vocab)
        self.vocab_size = len(self.vocab) + 1  # for padding +1

        self.df_ac_modeling_values = pd.DataFrame({'Anser_Tokens': ans_tokens_all,
                                                   'Anser_example_Tokens': ans_ex_tokens_all},
                                                  index = self.df_response_answer_ex.index)

        self.df_ac_modeling_values[self.dependent_var] = \
                 self.df_response_answer_ex[self.dependent_var]

        self.word_indices = dict((c, i + 1) for i, c in enumerate(self.vocab))

        self.ans_ex_maxlen = \
            max(map(len, (x for x in ans_ex_tokens_all)))
        self.ans_maxlen = \
            max(map(len, (x for x in ans_tokens_all)))
            
        # Modified by mack.sano@gmail.com 3/20/2020
        words = ["{word}\n".format(word=x) for x in self.vocab]
        with open( LOG_DIR + "/words.tsv", 'w', encoding="utf-8") as f:
            f. writelines(words)    

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            # Makoto.Sano@Mack-the-Psych.com
            with open(os.path.join(model_path, 'vdok3_memn2n.pkl'), 'rb') as inp:
                cls.model = pickle.load(inp)
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()        
        
        # Makoto.Sano@Mack-the-Psych.com
        memnd = tmv_tf_memn_classify_in('Independent_Variable_w_Label-Def.csv', 
                                         '/opt/program/vdok3/data/')
        memnd.load_data(input, dependent_var)        
        clf.perform_prediction(memnd.df_ac_modeling_values, number_class)
        
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
