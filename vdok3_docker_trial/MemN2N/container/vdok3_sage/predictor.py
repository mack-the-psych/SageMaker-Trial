# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

# Makoto.Sano@Mack-the-Psych.com
from tf_memn_classify import tmv_tf_memn_classify
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import numpy as np
import nltk.data

dependent_var = r'Definition-Score'
number_class = 3
csv_dump = False
batch_size = 100
epochs = 1
task_word = r'Definition'
key_word = r'TF_MEMN2N-Def-PRE-POST-All'

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

input_path = prefix + 'input/data'
channel_name='training'

'''
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
'''

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            # Makoto.Sano@Mack-the-Psych.com
            memnd = tmv_tf_memn_classify('/opt/program/vdok3/data/')
            
            memnd.load_data('Serialized-Def-ELVA.PILOT.PRE-TEST.csv', dependent_var, [0, 1], 
                        task_word)
            memnd.perform_modeling(memnd.df_ac_modeling_values, key_word, csv_dump, number_class, 
                                   epochs, batch_size)
            memnd.sess.close()
            
            saver = tf.train.Saver()
            sess = tf.Session()
            saver.restore(sess, os.path.join(model_path, 'vdok3_memn2n.ckpt'))
            memnd.sess = sess
            cls.model = memnd
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

        input = input.set_index(r'Student_Question_Index')
        ans_tokens = cls.get_tokens(input[answer_clm].values)
        ans_ex_tokens = cls.get_tokens(input[answer_ex_clm].values)

        df_ac_modeling_values = pd.DataFrame({'Anser_Tokens': ans_tokens,
                                                'Anser_example_Tokens': ans_ex_tokens})

        ans_tokens_vector = clf.vectorize_tokens(list(df_ac_modeling_values['Anser_Tokens']),
                                                  clf.ans_maxlen)
        ans_ex_tokens_vector = clf.vectorize_tokens(list(df_ac_modeling_values['Anser_example_Tokens']),
                                                  clf.ans_ex_maxlen)

        print(ans_tokens)
        print(ans_tokens_vector)
        print(ans_ex_tokens)
        print(ans_ex_tokens_vector)

        df_ac_predict_target = input.loc[:,[dependent_var]]
        y_test = df_ac_predict_target.transpose().values[0]
        y_matrix_test = y_test.reshape(len(y_test),1)
        ohe = OneHotEncoder(categorical_features=[0])
        y_ohe_test = ohe.fit_transform(y_matrix_test).toarray()

        answer_len = len(input)
        prediction = clf.sess.run(clf.y, feed_dict={
            clf.x: ans_ex_tokens_vector,
            clf.q: ans_tokens_vector,
            clf.a: y_ohe_test,
            clf.n_batch: answer_len
        })

        predict_res = np.zeros(answer_len, dtype=np.int)
        predict_res[0] =  np.argmax(prediction[0])

        clf.df_ac_classified = pd.DataFrame(np.array(predict_res, dtype=np.int64), None, 
                                            ['Score_Class'])
        clf.df_ac_classified.index.name = r'AC_Doc_ID'
        
        return clf.df_ac_classified

    @classmethod
    def get_tokens(cls, answer_str_list):
        list_cntnt = list(answer_str_list)
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

        tokens_all = []
        for x in list_cntnt:
            tokens = []
            sentences = sent_detector.tokenize(x.strip())
            for y in sentences:
                tokens += nltk.word_tokenize(y)
            tokens_all = tokens_all + [tokens]

        return tokens_all
    
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
