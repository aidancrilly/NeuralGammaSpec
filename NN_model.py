import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

class NNConvModel():
   
    def __init__(
            self,
            loss,
            model_file=None,
            weights_file=None,
            Adam_hyperparameters=
            {'learning_rate' : 0.0001,
             'beta_1' : 0.9,
             'beta_2' : 0.999},
            verbose=True):

        # Define the model architecture with Convolutional and Dense layers
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

        self.model = self._init_model()
        if weights_file is not None:
            self.model = self._load_weights() 
        if model_file is not None:
            self.model = self._load_model(model_file,loss)
        else:
            self.model.compile(optimizer=tf.keras.optimizers.Adam(**Adam_hyperparameters),
              loss=loss, metrics=['accuracy'])

        if verbose:
            print(self.model.summary())

    def _init_model(self):
        model = keras.Sequential()
        model.add(layers.Conv1D(64, 3, activation='relu', input_shape=(70, 1)))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Conv1D(128, 3, activation='relu'))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dense(1000))
        return model
    
    def _load_weights(self,file):
        self.model.load_weights(file)
    
    def _load_model(self,file,loss):
        model = keras.models.load_model(file, compile=True, custom_objects = {'loss' : loss})
        return model
    
    def fit(self,X,y,nepochs=20,verbose=False,validation_split=0.1, batch_size=32):
        history = self.fit(X, y, epochs=nepochs, verbose=verbose, validation_split=validation_split, batch_size=batch_size, callbacks=[self.callback])
        return history
    
    def save_model(self,file):
        self.model.save(file)

    def predict(self,X):
        return self.model.predict(tf.convert_to_tensor(X), verbose=0).reshape(-1)

class Ensemble_NNConvModel():

    def __init__(self,loss,model_files):
        self.models = []
        for model_file in model_files:
            self.models.append(NNConvModel(loss,model_file,verbose=False))
        self.ensemble_size = len(self.models)

    def predict(self,X):
        predictions = []
        for i in range(self.ensemble_size):
            predictOtherSequence = self.models[i].predict(X)

            predictions = predictions + [predictOtherSequence]

        predictions = np.array(predictions)
        meanpredictions = np.mean(predictions, axis=0)
        stdpredictions = np.std(predictions, axis=0)

        return meanpredictions, stdpredictions