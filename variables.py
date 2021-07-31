word2vec_path = 'weights_and_data/glove_vectors'

n_dim = 100
max_length = 20
trunc_type = 'post'
padding = 'post'
pad_token = '<pad>'
glove_path = 'GloveEmbedding/glove.6B/glove.6B.100d.txt'

encoder_dict_path = 'weights_and_data/label_encoder dict.pickle'
wordcloud_path = 'weights_and_data/wordcloud.png'
data_path = 'weights_and_data/Student Concerns_3.csv'

seed = 42
username = 'root'
password = 'root'
db_url = 'mysql+pymysql://{}:{}@localhost:3306/sms'.format(username,password)
table_name = 'student_concern_management'
