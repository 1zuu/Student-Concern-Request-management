import os
import re
import pickle
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from nltk.corpus import stopwords
from collections import defaultdict
from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import LabelEncoder

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

def preprocess_one(review):
    '''
        Text preprocess on term text using above functions
    '''
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords_list = stopwords.words('english')
    review = review.lower()
    remove_punc = tokenizer.tokenize(review) # Remove puntuations
    remove_num = [i for i in remove_punc if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_review = ' '.join(remove_stop)
    return updated_review

def preprocessed_data(reviews):
    '''
        Preprocess entire terms
    '''
    updated_reviews = []
    if isinstance(reviews, np.ndarray) or isinstance(reviews, list):
        for review in reviews:
            updated_review = preprocess_one(review)
            updated_reviews.append(updated_review)
    elif isinstance(reviews, np.str_)  or isinstance(reviews, str):
        updated_reviews = [preprocess_one(reviews)]

    return np.array(updated_reviews)

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

def get_data():
    df = pd.read_csv(data_path)
    del df['ID']
    df = df.dropna(axis=0)
    df = process_labels(df)

    student_concerns = df['Student Concern'].values
    processed_concerns = preprocessed_data(student_concerns)
    create_wordcloud(processed_concerns)

    departments = df['Department'].values
    sub_sections = df['Sub-section'].values
    concern_types = df['Concern Type'].values

    return processed_concerns, departments, sub_sections, concern_types

get_data()