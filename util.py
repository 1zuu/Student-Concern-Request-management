import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import pickle
import sqlalchemy
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

from variables import*

def word2vector():
    if not os.path.exists(word2vec_path):
        word2vec = {}
        with open(glove_path, encoding="utf8") as lines:
            for line in lines:
                line = re.split('[\n]', line)[0]
                line = line.split(' ')
                word, vec = line[0], line[1:]
                word = word.lower()
                word2vec[word] = np.array(list(map(float,vec)))

        file_ = open(word2vec_path,'wb')
        pickle.dump(word2vec, file_, protocol=pickle.HIGHEST_PROTOCOL)
        file_.close()
        print("Word2vec.pickle Saved!")
    else:
        print("Word2vec.pickle Loading!")
        file_ = open(word2vec_path,'rb')
        word2vec = pickle.load(file_)
        file_.close()
    return word2vec

def lemmatization(lemmatizer,sentence):
    '''
        Lemmatize texts in the terms
    '''
    lem = [lemmatizer.lemmatize(k) for k in sentence]
    lem = list(dict.fromkeys(lem))

    return [k for k in lem]

def remove_stop_words(stopwords_list,sentence):
    '''
        Remove stop words in texts in the terms
    '''
    return [k for k in sentence if k not in stopwords_list]

def preprocess_one(concern):
    '''
        Text preprocess on term text using above functions
    '''
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords_list = stopwords.words('english')
    concern = concern.lower()
    remove_punc = tokenizer.tokenize(concern) # Remove puntuations
    remove_num = [i for i in remove_punc if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_concern = ' '.join(remove_stop)
    return updated_concern

def preprocessed_data(concerns):
    '''
        Preprocess entire terms
    '''
    updated_concerns = []
    if isinstance(concerns, np.ndarray) or isinstance(concerns, list):
        for concern in concerns:
            updated_concern = preprocess_one(concern)
            updated_concerns.append(updated_concern)
    elif isinstance(concerns, np.str_)  or isinstance(concerns, str):
        updated_concerns = [preprocess_one(concerns)]

    return np.array(updated_concerns)

def process_labels(df):
    df_labels = df[['Department', 'Sub-section', 'Concern Type']]
    df_labels = df_labels.apply(lambda x: x.astype(str).str.lower())
    df_labels = df_labels.apply(lambda x: x.astype(str).str.strip())

    df_labels = label_encoding(df_labels)
    df[['Department', 'Sub-section', 'Concern Type']] = df_labels
    return df


def label_encoding(df_cat):
    if not os.path.exists(encoder_dict_path):
        encoder_dict = defaultdict(LabelEncoder)
        encoder = df_cat.apply(lambda x: encoder_dict[x.name].fit_transform(x))
        encoder.apply(lambda x: encoder_dict[x.name].inverse_transform(x))
        with open(encoder_dict_path, 'wb') as handle:
            pickle.dump(encoder_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(encoder_dict_path, 'rb') as handle:
            encoder_dict = pickle.load(handle)
    return df_cat.apply(lambda x: encoder_dict[x.name].transform(x))

def create_wordcloud(processed_concerns):
    long_string = ','.join(list(processed_concerns))
    wordcloud = WordCloud(
                        width=1600, 
                        height=800, 
                        max_words=200, 
                        background_color='white',
                        max_font_size=200, 
                        random_state=seed
                        )
    wordcloud.generate(long_string)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("WordCloud Distribution of Student Concerns")
    plt.savefig(wordcloud_path)
    plt.show()

def derive_vocabulary(processed_concerns):
    '''
        Derive the vocabulary from the processed concerns
    '''
    vocabulary = {}
    lengths = []
    for concern in processed_concerns:
        concern = concern.split(' ')
        lengths.append(len(concern))
        for word in concern:
            word = word.strip()
            if word not in vocabulary:
                vocabulary[word] = 1
            else:
                vocabulary[word] += 1

    vocabulary = {k: v for k, v in sorted(
                                        vocabulary.items(), 
                                        key=lambda item: item[1],
                                        reverse=True)}
    word2index = {k: i+1 for i, (k, v) in enumerate(vocabulary.items())}
    word2index[pad_token] = 0
    
    return word2index

def sequence_and_padding_concerns(processed_concerns, word2index):
    seq_concerns = []
    for concern in processed_concerns:
        seq_concern = []
        concern = concern.split(' ')
        for word in concern:
            word = word.strip() 
            seq_concern.append(word2index[word])
        seq_concerns.append(seq_concern)

    pad_concerns = pad_sequences(
                            seq_concerns, 
                            maxlen=max_length, 
                            truncating=trunc_type,
                            padding=padding
                            )
    pad_concerns = np.array(pad_concerns)
    return pad_concerns

def word_embeddings(pad_concerns, word2index):
    N = pad_concerns.shape[0]
    embedding_concerns = np.zeros((N, max_length, n_dim))

    index2word = {v:k for k,v in word2index.items()}

    word2vec = word2vector()
    for i, concern in enumerate(pad_concerns):
        for j, index in enumerate(concern):
            word = index2word[index]
            if (word not in word2vec) and (word != pad_token):
                embedding_concerns[i,j,:] = word2vec['unk']
            else:
                if word != pad_token: 
                    embedding_concerns[i,j,:] = word2vec[word]
    return embedding_concerns

def create_database(engine):
    engine = create_engine(db_url)
    if table_name not in sqlalchemy.inspect(engine).get_table_names():
        data = pd.read_csv(data_path, encoding= 'unicode_escape')
        data = data.dropna(axis=0)
        with engine.connect() as conn, conn.begin():
            data.to_sql(table_name, conn, if_exists='append', index=False)

def get_data():
    engine = create_engine(db_url)
    create_database(engine)
    data = pd.read_sql_table(table_name, engine)
    data = process_labels(data)

    student_concerns = data['Student Concern'].values
    processed_concerns = preprocessed_data(student_concerns)
    
    word2index = derive_vocabulary(processed_concerns)
    pad_concerns = sequence_and_padding_concerns(processed_concerns, word2index)
    embedding_concerns = word_embeddings(pad_concerns, word2index)

    create_wordcloud(processed_concerns)

    departments = data['Department'].values
    sub_sections = data['Sub-section'].values
    concern_types = data['Concern Type'].values

    return embedding_concerns, departments, sub_sections, concern_types

get_data()