# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function
from RF_classify import tmv_RF_classify

import os
import json
import pickle
import io as StringIO
import sys
import signal
import traceback
import flask
import pandas as pd
import numpy as np

from qa_serializer_lang_selector import qa_serializer_lang_selector
from basic_nlp import fex_basic_nlp
from bi_trigram import bi_trigram
from oanc_lemma_frequency import odi_oanc_lemma_frequency
from overlapping import odi_overlapping
import ac_bi_trigram_pmi_distribution as gpmd
import ac_aggregate_plim as agpl
import ac_aggregate_item_level_plim as agpi

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
oanc_resource = '/opt/program/plimac3/Resource/OANC/'
data_dir = '/opt/program/vdok3/data/'
data_file = 'vdok_predction_data_file.csv'
def_file = 'Questin_ID_Definition.csv'
pmi_bigram_file = 'PMI-Sum-T-Bigram-Def-PRE.csv'
pmi_trigram_file = 'PMI-Sum-T-Trigram-Def-PRE.csv'

dependent_var = 'Definition-Score'
drop_vars = None
task_name = 'Definition'
stop_words = ['a', 'be', 'to', 'and', 'or']
stop_words_hy = ['be']
stop_words_pos = None
specific_count_lemmas = ['dk', 'nr']

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
    def remove_non_extracted_stop_word(cls, df_ac, stop_words):
        stop_words_buf = stop_words[:]
        for x in stop_words:
            if not df_ac.columns.isin([x]).any():
                print('Remove Stop Word: ', x)
                stop_words_buf.remove(x)
        return stop_words_buf

    @classmethod
    def bi_trigram_pmi_distribution(cls, csv_file_pmi_sum_t, data_dir, num_clm_in_q, df_ac_gram, 
                                    gram = 'bigram', pmi_frq_min = 2, decimal_places = 4):
        df_ac_pmi_sum_t = pd.read_csv(data_dir + csv_file_pmi_sum_t, encoding= 'latin1')
        if gram == 'bigram':
            sum_clm = 'Bigram_sum'
        else: sum_clm = 'Trigram_sum'

        df_ac_pmi_gram = df_ac_pmi_sum_t[df_ac_pmi_sum_t[sum_clm] >= pmi_frq_min]
        df_ac_pmi_dist_gram = gpmd.ac_bi_trigram_pmi_distribution(df_ac_gram,
                        num_clm_in_q + 1, df_ac_pmi_gram, gram, decimal_places)
        return df_ac_pmi_dist_gram

    @classmethod
    def aggregate_plim(cls, bnlqd, oalqd, ovlqd, df_ac_pmi_dist_bigram, df_ac_pmi_dist_trigram, bnlpd,
                             specific_count_lemmas = None, stop_words_pos = None, 
                             task_name = 'Definition', decimal_places = 4):
        stem_identifier = task_name + '-Question'
        option_identifier = task_name + '-Answer'
        df_ac_lemma_buf = bnlqd.df_ac_lemma.copy()

        if specific_count_lemmas is not None:
            for x in specific_count_lemmas:
                if not bnlqd.df_ac_lemma.columns.isin([x]).any():
                    df_ac_lemma_buf[x] = 0

        df_ac_oanc_lemma_freq = oalqd.df_ac_oanc_lemma_freq_q.drop([oalqd.question_id_clm,
                                 oalqd.stem_option_name_clm], axis=1)
        df_ac_overlapping_lemma = ovlqd.df_ac_overlapping_lemma.drop([oalqd.question_id_clm,
                                 oalqd.stem_option_name_clm], axis=1)
        df_ac_overlapping_synset = ovlqd.df_ac_overlapping_syn_lemma.drop([oalqd.question_id_clm,
                                 oalqd.stem_option_name_clm], axis=1)
        df_ac_overlapping_hyper = ovlqd.df_ac_overlapping_hyper_lemma.drop([oalqd.question_id_clm,
                                 oalqd.stem_option_name_clm], axis=1)
        df_ac_overlapping_hypo = ovlqd.df_ac_overlapping_hypo_lemma.drop([oalqd.question_id_clm,
                                 oalqd.stem_option_name_clm], axis=1)

        df_ac_pmi_dist_bigram = df_ac_pmi_dist_bigram.iloc[:, oalqd.num_clm_in_q:]
        df_ac_pmi_dist_bigram['Cntnt_Bigram'] = df_ac_pmi_dist_bigram['Cntnt_Bigram'].fillna('')
        df_ac_pmi_dist_bigram['PMI_Bigram_SD'] = df_ac_pmi_dist_bigram['PMI_Bigram_SD'].fillna(0.0)
        df_ac_pmi_dist_bigram = df_ac_pmi_dist_bigram.fillna(-10.0)

        df_ac_pmi_dist_trigram = df_ac_pmi_dist_trigram.iloc[:, oalqd.num_clm_in_q:]
        df_ac_pmi_dist_trigram['Cntnt_Trigram'] = df_ac_pmi_dist_trigram['Cntnt_Trigram'].fillna('')
        df_ac_pmi_dist_trigram['PMI_Trigram_SD'] = df_ac_pmi_dist_trigram['PMI_Trigram_SD'].fillna(0.0)
        df_ac_pmi_dist_trigram = df_ac_pmi_dist_trigram.fillna(-10.0)

        if specific_count_lemmas == None:
            df_ac_lemma = None
        else:
            df_ac_lemma = df_ac_lemma_buf

        df_ac_aggregate = agpl.ac_aggregate_plim(bnlqd.df_ac_pos, oalqd.num_clm_in_q + 1, 
                                df_ac_overlapping_lemma, df_ac_overlapping_synset, 
                                None, df_ac_oanc_lemma_freq, oalqd.stem_option_name_clm, stem_identifier,
                                list(oalqd.df_ac_in_q.columns), stop_words_pos, df_ac_lemma,
                                specific_count_lemmas, bnlpd.df_ac_pos, ovlqd.passage_name_clm_q,
                                ovlqd.passage_sec_clm_q, ovlqd.passage_name_clm_p, ovlqd.passage_sec_clm_p,
                                bnlpd.num_clm_in + 1, decimal_places,
                                df_ac_overlapping_hyper, df_ac_overlapping_hypo,
                                df_ac_bigram_pmi_distribution = df_ac_pmi_dist_bigram, 
                                df_ac_trigram_pmi_distribution = df_ac_pmi_dist_trigram)

        key_dummy = 'Key_Dummy'
        t = df_ac_aggregate.shape
        row_lgth = t[0]
        df_key_dummy = pd.DataFrame(np.empty((row_lgth, 1),
                            dtype=object), df_ac_aggregate.index,
                            [key_dummy])
        df_key_dummy = df_key_dummy.fillna(option_identifier)
        df_ac_aggregate[key_dummy] = df_key_dummy[key_dummy]

        return df_ac_aggregate

    @classmethod
    def aggregate_item_level_plim(cls, df_ac_aggregate, stem_option_name_clm, task_name = 'Definition', 
                                  cntnt_clm = 'Content', decimal_places = 4):
        stem_identifier = task_name + '-Question'
        df_ac_aggregate_item_level = agpi.ac_aggregate_item_level_plim(df_ac_aggregate,
                                'Key_Dummy', stem_option_name_clm, stem_identifier, 
                                None, decimal_places, cntnt_clm)
        return df_ac_aggregate_item_level

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()        

        input.to_csv(data_dir + 'vdok_predction_src_file.csv')

        q = qa_serializer_lang_selector(data_dir)
        q.serialize_record('vdok_predction_src_file.csv', task_name)
        q.select_lang([1], task_name).to_csv(data_dir + data_file, encoding= 'latin1')

        pipeline=['pos', 'lemma', 'synset', 'hype', 'hypo']

        bnlqd = fex_basic_nlp(data_file, data_dir)
        bnlqd.nlp_run(pipeline[0])
        bnlqd.nlp_run(pipeline[1])
        bnlqd.df_ac_lemma.to_csv(data_dir + 'Lemma-' + data_file, encoding= 'latin1')
        bnlqd.nlp_run(pipeline[2])
        bnlqd.df_ac_synset.to_csv(data_dir + 'Synset-' + data_file , encoding= 'latin1')
        bnlqd.nlp_run(pipeline[3])
        bnlqd.df_ac_hypernyms.to_csv(data_dir + 'Hypernyms-' + data_file, encoding= 'latin1')
        bnlqd.nlp_run(pipeline[4])
        bnlqd.df_ac_hyponyms.to_csv(data_dir + 'Hyponyms-' + data_file, encoding= 'latin1')

        bnlpd = fex_basic_nlp(def_file, data_dir, task_name)
        bnlpd.nlp_run(pipeline[0])
        bnlpd.nlp_run(pipeline[1])
        bnlpd.df_ac_lemma.to_csv(data_dir + 'Lemma-P-' + data_file, encoding= 'latin1')
        
        btgqd = bi_trigram(data_file, data_dir)
        btgqd.nlp_run(r'bigram')
        btgqd.nlp_run(r'trigram')         

        stop_words_d = cls.remove_non_extracted_stop_word(bnlqd.df_ac_lemma, stop_words)

        oanc_shelve = oanc_resource + 'ANC-all-lemma-04262014.db'
        oalqd = odi_oanc_lemma_frequency(data_file, oanc_shelve, None, data_dir, stop_words_d)    
        oalqd.oanc_lemma_frequency('Lemma-' + data_file, 'Student_Question_Index', 'Pre_Col_Name')
        
        stop_words_hy_d = cls.remove_non_extracted_stop_word(bnlqd.df_ac_lemma, stop_words_hy)

        ovlqd = odi_overlapping(data_file, def_file, data_dir, stop_words_d)
        ovlqd.count_overlapping('Lemma-' + data_file, 'Student_Question_Index',
                             'Pre_Col_Name', 'Question_ID', 'Question_ID_Sec',
                             'Lemma-P-' + data_file, 'Question_ID', 'Question_ID_Sec')
        ovlqd.count_overlapping_synset('Synset-' + data_file)
        ovlqd.count_overlapping_hypernyms('Hypernyms-' + data_file, stop_words_hy_d)
        ovlqd.count_overlapping_hyponyms('Hyponyms-' + data_file, stop_words_hy_d)

        df_ac_pmi_dist_bigram = cls.bi_trigram_pmi_distribution(pmi_bigram_file, data_dir, 
                                                            bnlqd.num_clm_in, btgqd.df_ac_bigram, 'bigram')
        df_ac_pmi_dist_trigram = cls.bi_trigram_pmi_distribution(pmi_trigram_file, data_dir, 
                                                            bnlqd.num_clm_in, btgqd.df_ac_trigram, 'Trigram')

        df_ac_aggregate = cls.aggregate_plim(bnlqd, oalqd, ovlqd, df_ac_pmi_dist_bigram, df_ac_pmi_dist_trigram,
                bnlpd, specific_count_lemmas, stop_words_pos, task_name)
        df_ac_aggregate.to_csv(data_dir + 'vdok_predction_Aggregate_plim.csv', encoding= 'latin1')
        df_ac_aggregate_item_level = cls.aggregate_item_level_plim(df_ac_aggregate, oalqd.stem_option_name_clm, 
                                                                   task_name)
        df_ac_aggregate_item_level.to_csv(data_dir + 'vdok_predction_Key_Stem_Passage_Aggregate_plim.csv',
                                          encoding= 'latin1')

        rfrpod = tmv_RF_classify('Independent_Variable_w_Label-Def.csv', data_dir)
        rfrpod.load_data('vdok_predction_Key_Stem_Passage_Aggregate_plim.csv', True, drop_vars, dependent_var)
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
