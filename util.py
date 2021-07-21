import os
import re
import pickle
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

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
        pickle.dump(word2vec, file_)
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
        Lematize texts in the terms
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

def get_data():
    df = pd.read_csv(data_path)
    df = df.dropna(axis=0)
    student_concerns = df['student_concerns'].values
    print(student_concerns)
