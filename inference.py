import os

from sklearn import neighbors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import pickle
import sqlalchemy
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing.sequence import pad_sequences

from dnn import SCRM_Model
from variables import*
from util import*

class SCRM_Inference(object):
    def __init__(self):
        self.engine = create_engine(db_url)
        self.scrm = SCRM_Model()
        self.scrm.run()
        self.vocab = self.scrm.word2index

    def predict(self, embedding_concerns):
        output = []
        for embedding_concerns in embedding_concerns:
            vector = self.scrm.runTFliteInference(embedding_concerns, False)
            output.append(vector)

        outputs = np.array(output)
        return outputs

    def data_to_features(self):
        if not os.path.exists(data_feature_path):
            data = pd.read_sql_table(table_name, self.engine)
            data = data.dropna(axis=0)

            student_concerns = data['Student_Concern'].values
            solutions = data['Solution'].values
            student_id = data['id'].values
            embedding_concerns = text_processing(student_concerns)[1]

            features = self.predict(embedding_concerns)
            np.savez(
                    data_feature_path, 
                    name1=features, 
                    name2=solutions, 
                    name3=student_id
                    )
        else:
            data = np.load(data_feature_path)
            features = data['name1']
            solutions = data['name2']
            student_id = data['name3']

        self.features = features
        self.solutions = solutions
        self.student_id = student_id

    def nearest_neighbor_model(self):
        if not os.path.exists(n_neighbour_weights):
            # self.test_features = self.test_features.reshape(self.test_features.shape[0],-1)
            self.neighbor = NearestNeighbors(
                                        n_neighbors = n_neighbors,
                                        )
            self.neighbor.fit(self.features)
            with open(n_neighbour_weights, 'wb') as file:
                pickle.dump(self.neighbor, file)
        else:
            with open(n_neighbour_weights, 'rb') as file:
                self.neighbor = pickle.load(file)

    def process_inf_data(self, concern):
        preprocessed_concern = preprocess_one(concern)
        pad_concerns = sequence_and_padding_concerns([preprocessed_concern], self.vocab)
        embedding_concern = word_embeddings(pad_concerns, self.vocab)
        embedding_concern = embedding_concern.reshape(max_length, embedding_dim)
        return embedding_concern

    def predict_one(self, concern):
        embedding_concern = self.process_inf_data(concern)
        feature_vector = self.scrm.runTFliteInference(embedding_concern, False).squeeze()
        categories = self.scrm.runTFliteInference(embedding_concern).squeeze()
        return feature_vector, categories

    def predict_best_solution(self, concern):
        feature_vector, categories = self.predict_one(concern)
        neighbor = self.neighbor.kneighbors([feature_vector], 1)[1]
        neighbor = neighbor.squeeze()
        return self.solutions[neighbor]

inf = SCRM_Inference()
inf.data_to_features()
inf.nearest_neighbor_model()

concern = 'Can I join to mulitiple sports teams? '
solution = inf.predict_best_solution(concern)

print("concern  : {} ".format(concern))
print("solution : {} ".format(solution))