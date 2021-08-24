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
from tensorflow.keras.activations import relu

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
        embedding_concerns, outputs, word2index = get_data()
        self.X = embedding_concerns
        self.Y = outputs
        self.word2index = word2index

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
        x = Bidirectional(
                    LSTM(
                       size_lstm1,
                       return_sequences=True,
                       unroll=True
                       ), name='bidirectional_lstm1')(inputs) # Bidirectional LSTM layer
        x = Bidirectional(
                    LSTM(
                       size_lstm2,
                       unroll=True
                       ), name='bidirectional_lstm2')(x) # Bidirectional LSTM layer
                       
        x = Dense(dense1, activation='relu')(x)
        x = Dense(dense1)(x) 
        x = BatchNormalization()(x)
        x = relu(x)
        x = Dropout(keep_prob)(x)

        x = Dense(dense2, activation='relu')(x) 
        x = Dense(dense2)(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Dropout(keep_prob)(x)

        x = Dense(dense3, activation='relu')(x) 
        x = Dense(dense3, name='features')(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Dropout(keep_prob)(x)

        output1 = Dense(output_dim1, activation='softmax', name='Department')(x)
        output2 = Dense(output_dim2, activation='softmax', name='Sub_Section')(x)
        output3 = Dense(output_dim3, activation='softmax', name='Concern_Type')(x)

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
                                [self.Y[:,i] for i in range(self.Y.shape[1])],
                                validation_split = val_size,
                                batch_size=batch_size,
                                epochs=num_epochs,
                                verbose=2
                                )

    def run_classifier(self):
        self.classifier()
        self.train()

    def feature_extraction(self):
        inputs = self.model.input
        outputs = self.model.layers[-5].output
        feature_model = Model(
                        inputs =inputs,
                        outputs=outputs
                            )
        self.feature_model = feature_model
        self.feature_model.summary()

    def save_model(self, train=True): # Save trained model
        if train:
            self.model.save(model_weights)
        else:
            self.feature_model.save(fmodel_weights)

    def loading_model(self, train=True): # Load and compile pretrained model
        if train:
            self.model = load_model(model_weights, compile=False)
            self.model.compile(
                            loss='sparse_categorical_crossentropy', 
                            optimizer='adam', 
                            metrics=['accuracy']
                            )
        else:
            self.feature_model = load_model(fmodel_weights)

    def TFconverter(self, model_path, converter_path):
        # converter = tf.lite.TFLiteConverter.from_keras_model(self.feature_model)
        converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_path) 
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        model_converter_file = pathlib.Path(converter_path)
        model_converter_file.write_bytes(tflite_model)

    def TFinterpreter(self, converter_path):
        # Load the TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=converter_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details

    def TFliteInference(self, description, interpreter, input_details, output_details, train):
        input_shape = input_details[0]['shape']
        input_data = np.expand_dims(description, axis=0).astype(np.float32)
        assert np.array_equal(input_shape, input_data.shape), "required shape : {} doesn't match with provided shape : {}".format(input_shape, input_data.shape)

        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        if train:
            concern_type = interpreter.get_tensor(output_details[0]['index']) #Concern_Type
            department = interpreter.get_tensor(output_details[1]['index']) #Department
            subsection = interpreter.get_tensor(output_details[2]['index']) #Sub_Section

            concern_type = concern_type.squeeze().argmax(axis=-1)
            department = department.squeeze().argmax(axis=-1)
            subsection = subsection.squeeze().argmax(axis=-1)
            
            output_data = np.array([department, subsection, concern_type])

        else:
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
        return output_data

    def runTFconverter(self):
        self.TFconverter(model_weights, model_converter)
        self.TFconverter(fmodel_weights, fmodel_converter)

    def runTFinterpreter(self):
        interpreter, input_details, output_details = self.TFinterpreter(model_converter)
        self.model_interpreter = interpreter
        self.model_input_details = input_details
        self.model_output_details = output_details

        interpreter, input_details, output_details = self.TFinterpreter(fmodel_converter)
        self.fmodel_interpreter = interpreter
        self.fmodel_input_details = input_details
        self.fmodel_output_details = output_details

    def runTFliteInference(self, description, train=True):
        if train:
            interpreter = self.model_interpreter
            input_details = self.model_input_details
            output_details = self.model_output_details
        else:
            interpreter = self.fmodel_interpreter
            input_details = self.fmodel_input_details
            output_details = self.fmodel_output_details
        return self.TFliteInference(description, interpreter, input_details, output_details, train)

    def run(self):
        if not os.path.exists(model_converter):
            if not os.path.exists(model_weights):
                self.run_classifier()
                self.save_model()

                self.feature_extraction()
                self.save_model(False)
            else:
                self.loading_model()
                self.loading_model(False)
            self.runTFconverter()
        self.runTFinterpreter()