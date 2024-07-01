from keras.src.models import Sequential
from keras.src.models import Sequential
from keras.src.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.src.layers import Normalization, Reshape,GlobalMaxPooling1D
from keras.src.optimizers import Adam,Adadelta
import tensorflow as tf
from keras import models,saving
from keras import regularizers
from train_cnn import split_dataset



def create_model():
    max_token = 1081

    model = Sequential([  
    Embedding(input_dim=20000, output_dim=128, input_length=max_token),
    Conv1D(filters=32, kernel_size=5, activation='relu'),
    BatchNormalization(),  
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    BatchNormalization(), 
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
    ])


    model.compile(loss='binary_crossentropy', 
              optimizer=Adam(learning_rate=0.001)) 

    model.build(input_shape=(None, 1081))

    saving.save_model(model,"data/model/model.keras")



if __name__ == '__main__':
    create_model()
