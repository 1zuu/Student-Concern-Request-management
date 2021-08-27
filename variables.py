word2vec_path = 'data/glove_vectors.pickle'
glove_path = 'data/GloveEmbedding/glove.6B/glove.6B.100d.txt'

model_weights = 'weights/model_weights.h5'
model_converter = 'weights/model_converter.tflite'

fmodel_weights = 'weights/feature_model_weights.h5'
fmodel_converter = 'weights/feature_model_converter.tflite'

data_feature_path = 'weights/features.pickle'
vocabulary_path = 'weights/vocabulary.pickle'

n_neighbour_weights = 'weights/nearest_neighbour.pickle'
encoder_dict_path = 'weights/label_encoder dict.pickle'
wordcloud_path = 'data/wordcloud.png'
data_path = 'data/Student Concerns_5.csv'

seed = 42
host = '127.0.0.1'
port = 27017
database = 'SCRMS'
db_collection = 'concerns'
live_collection = 'live_concerns'
username = 'root'
password = 'root'
# db_url = 'mongodb://{}:{}@{}:{}/{}?authSource=admin'.format(username, password, host, port, database)
db_url = "mongodb://localhost:27017/"

vocab_size = 1000
max_length = 30
embedding_dim = 100
trunc_type = 'post'
padding = 'post'
oov_tok = "<oov>"
pad_token = '<pad>'
num_epochs = 20
batch_size = 32
size_lstm1  = 256
size_lstm2  = 128
dense1 = 256
dense2 = 128
dense3 = 64
keep_prob = 0.4

test_size = 0.005
val_size = 0.15
n_neighbors = 1