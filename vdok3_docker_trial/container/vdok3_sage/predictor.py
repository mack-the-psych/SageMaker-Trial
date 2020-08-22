# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function
from RF_classify import tmv_RF_classify

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

class tmv_RF_classify_df_in(tmv_RF_classify):
    def load_data(self, df_in, labeled = True, drop_ind_vars = None, dependent_var = None):
        self.dependent_var = dependent_var
        # Makoto.Sano@Mack-the-Psych.com
        self.df_ac_aggregate_item_level = df_in
        self.df_ac_aggregate_item_level = self.df_ac_aggregate_item_level.set_index('AC_Doc_ID')
        self.df_ac_modeling_values = self.df_ac_aggregate_item_level.loc[:,
                                                        list(self.df_ind_vars['Variables'])]
        if labeled == True:
            for x in self.df_ac_modeling_values.columns:
                if x in self.df_ind_vars['Variables'].values:
                    self.df_ac_modeling_values = self.df_ac_modeling_values.rename(
                        columns={x : self.df_ind_vars[self.df_ind_vars['Variables'].isin([x])]['Label'].values[0]})

        if drop_ind_vars != None:
            self.df_ac_modeling_values = self.df_ac_modeling_values.drop(drop_ind_vars, axis=1)
                    
        if self.dependent_var != None:
            self.df_ac_modeling_values[self.dependent_var] = \
                    self.df_ac_aggregate_item_level[self.dependent_var]

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            with open(os.path.join(model_path, 'vdok3_rf.pkl'), 'rb') as inp:
                cls.model = pickle.load(inp)
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()        
        
        rfrpod = tmv_RF_classify_df_in('Independent_Variable_w_Label-Def.csv', '/opt/program/vdok3/data/')
        rfrpod.load_data(input, True, drop_vars, dependent_var)
        clf.perform_prediction(rfrpod.df_ac_modeling_values)
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
