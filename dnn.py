import os
import pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import warnings
import numpy as np
import tensorflow as tf
logging.getLogger('tensorflow').disabled = True
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model

from variables import *
from util import*

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\n Num GPUs Available: {}\n".format(len(physical_devices)))
if len(physical_devices):
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

np.random.seed(seed)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True
warnings.simplefilter("ignore", DeprecationWarning)

class SCRM_Model():
    def __init__(self):
        embedding_concerns, outputs = get_data()
        self.X = embedding_concerns
        self.Y = outputs

        X, Xtest, Y, Ytest = train_test_split(
                                            self.X, 
                                            self.Y, 
                                            test_size=test_size, 
                                            random_state=seed
                                            )
        self.X = X
        self.Y = Y
        self.Xtest = Xtest
        self.Ytest = Ytest

    def classifier(self): # Building the RNN model
        output_dim1 = len(set(self.Y[:,0]))
        output_dim2 = len(set(self.Y[:,1]))
        output_dim3 = len(set(self.Y[:,2]))

        inputs = Input(shape=(max_length,embedding_dim))
        x = Bidirectional(LSTM(size_lstm), name='bidirectional_lstm')(inputs) # Bidirectional LSTM layer
        x = Dense(dense1, activation='relu')(x)
        x = Dense(dense1, activation='relu')(x) 
        x = Dropout(keep_prob)(x)
        x = Dense(dense2, activation='relu')(x) 
        x = Dense(dense2, activation='relu')(x)
        x = Dropout(keep_prob)(x)
        x = Dense(dense3, activation='relu')(x) 
        x = Dense(dense3, activation='relu')(x)
        x = Dropout(keep_prob)(x)

        output1 = Dense(output_dim1, activation='softmax')(x)
        output2 = Dense(output_dim2, activation='softmax')(x)
        output3 = Dense(output_dim3, activation='softmax')(x)

        model = Model(
                   inputs = inputs, 
                   outputs = [output1, output2, output3]
                     )
        self.model = model

    def train(self): # Compile the model and training
        self.model.compile(
                        loss='sparse_categorical_crossentropy', 
                        optimizer='adam', 
                        metrics=['accuracy']
                        )
        self.model.summary()
        self.history = self.model.fit(
                                self.X,
                                [self.Y[:,0], self.Y[:,1], self.Y[:,2]],
                                # validation_data = [self.X_pad_test, self.Ytest],
                                # batch_size=batch_size,
                                epochs=num_epochs
                                )

    def run_classifier(self):
        self.classifier()
        self.train()

    def feature_extraction_model(self):
        inputs = self.model.input
        outputs = self.model.layers[-4].output
        feature_model = Model(
                        inputs =inputs,
                        outputs=outputs
                            )
        self.feature_model = feature_model
        self.feature_model.summary()

    def save_model(self): # Save trained model
        self.feature_model.save(model_weights)

    def loading_model(self): # Load and compile pretrained model
        self.feature_model = load_model(model_weights)
        self.feature_model.compile(
                                loss='sparse_categorical_crossentropy', 
                                optimizer='adam', 
                                metrics=['accuracy']
                                    )

    def TFconverter(self):
        # converter = tf.lite.TFLiteConverter.from_keras_model(self.feature_model)
        converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_weights)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        model_converter_file = pathlib.Path(model_converter)
        model_converter_file.write_bytes(tflite_model)

    def TFinterpreter(self):
        # Load the TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=model_converter)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def Inference(self, img):
        input_shape = self.input_details[0]['shape']
        input_data = np.expand_dims(img, axis=0).astype(np.float32)
        assert np.array_equal(input_shape, input_data.shape), "Input tensor hasn't correct dimension"

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

    def extract_features(self):
        if not os.path.exists(n_neighbour_weights):
            self.test_features = np.array(
                            [self.Inference(img) for img in self.test_images]
                                        )
            self.test_features = self.test_features.reshape(self.test_features.shape[0],-1)
            self.neighbor = NearestNeighbors(
                                        n_neighbors = 20,
                                        )
            self.neighbor.fit(self.test_features)
            with open(n_neighbour_weights, 'wb') as file:
                pickle.dump(self.neighbor, file)
        else:
            with open(n_neighbour_weights, 'rb') as file:
                self.neighbor = pickle.load(file)

    def run(self):
        if not os.path.exists(model_converter):
            if not os.path.exists(model_weights):
                self.run_classifier()
                self.feature_extraction_model()
                self.save_model()
            else:
                self.loading_model()
            self.TFconverter()
        self.TFinterpreter()    
        # self.extract_features()

if __name__ == "__main__":

    if not os.path.exists(os.path.join(os.getcwd(), 'weights_and_data')):
        os.makedirs(os.path.join(os.getcwd(), 'weights_and_data'))
    model = SCRM_Model()
    model.run()