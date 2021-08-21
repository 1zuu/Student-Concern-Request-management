word2vec_path = 'weights_and_data/glove_vectors.pickle'
glove_path = 'GloveEmbedding/glove.6B/glove.6B.100d.txt'

model_weights = 'weights_and_data/model_weights.h5'
model_converter = 'weights_and_data/model_converter.h5'
n_neighbour_weights = 'weights_and_data/nearest_neighbour.pickle'
encoder_dict_path = 'weights_and_data/label_encoder dict.pickle'
wordcloud_path = 'weights_and_data/wordcloud.png'
data_path = 'weights_and_data/Student Concerns_4.csv'

seed = 42
username = 'root'
password = 'root'
db_url = 'mysql+pymysql://{}:{}@localhost:3306/scms'.format(username,password)
table_name = 'concerns'

vocab_size = 1000
max_length = 30
embedding_dim = 100
trunc_type = 'post'
padding = 'post'
oov_tok = "<oov>"
pad_token = '<pad>'
num_epochs = 20
batch_size = 64
size_lstm  = 256
dense1 = 256
dense2 = 128
dense3 = 64
keep_prob = 0.4

test_size = 0.02