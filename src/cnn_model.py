from keras.src.models import Sequential
from keras.src.models import Sequential
from keras.src.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.src.layers import Normalization, Reshape,GlobalMaxPooling1D
from keras.src.optimizers import Adam,Adadelta
import tensorflow as tf
from keras import models,saving
from keras import regularizers
from train_cnn import split_dataset



def create_model(x_train):
    # model = Sequential()   //onaj osnovni
    # input_dim=x_train.shape[1]
    # model.add(Dense(10,input_dim=input_dim,activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))

    #model = Sequential()
    max_token = 1081

    # model = Sequential([ # create a layer-stacked sequential neural network model   sa jednim conv oko 91,92 a sa dva 89
    #     Embedding(2000, (128), input_length=1081), # create embedding layer to give close values to tokens that are similar
    #     Conv1D(16, 5, activation='relu'), # create convolution layer for batch learning to better identify features
    #     Conv1D(filters=32, kernel_size=5, activation='relu'),
    #     GlobalMaxPooling1D(), # create global pooling layer to return only the maximum value of each batch to further emphasize characteristics
    #     Dense(1, activation='sigmoid') # create dense layer with one unit and apply activation by sigmoidal function
    #     ])

   

    # model = Sequential([    odlican oko 91,92,93
    #     Embedding(input_dim=2000, output_dim=128, input_length=max_token),
    #     Conv1D(filters=16, kernel_size=5, activation='relu'),
    #     MaxPooling1D(pool_size=2),  # Added MaxPooling1D layer
    #     Dropout(0.5),  # Added Dropout layer
    #     Conv1D(filters=32, kernel_size=5, activation='relu'),
    #     MaxPooling1D(pool_size=2),  # Added MaxPooling1D layer
    #     Dropout(0.5),  # Added Dropout layer
    #     GlobalMaxPooling1D(),  # Added GlobalMaxPooling1D layer
    #     Dense(128, activation='relu'),  # Dense layer for learning complex features
    #     Dropout(0.5),  # Added Dropout layer
    #     Dense(64, activation='relu'),  # Dense layer for further feature learning
    #     Dropout(0.5),  # Added Dropout layer
    #     Dense(1, activation='sigmoid')  # Output layer for binary classification
    # ])


    model = Sequential([    #sa normalizacijom
    Embedding(input_dim=2000, output_dim=128, input_length=max_token),
    Conv1D(filters=16, kernel_size=5, activation='relu'),
    BatchNormalization(),  # Apply BatchNormalization
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Conv1D(filters=32, kernel_size=5, activation='relu'),
    BatchNormalization(),  # Apply BatchNormalization
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
    ])

    # model.add(Normalization())
    # model.add(Embedding(input_dim=len(x_train), output_dim=256, input_length=sequence_length))    #jovanin
    # model.add(Conv1D(128, 7, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling1D())
    # model.add(Dropout(0.5))
    # model.add(Conv1D(256, 5, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling1D())
    # model.add(Dropout(0.5))
    # model.add(Conv1D(512, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))

    # Reshape input data for 1D convolution
    # model.add(Reshape((sequence_length, 1), input_shape=(sequence_length,)))  # Adjust input_shape accordingly  //prvo smo ovo i sve ispod


    #model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length))   #dodato ovo umesto reshape i l2 dole

    # # Add Conv1D layers
    # model.add(Conv1D(128, 7, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling1D())
    # model.add(Dropout(0.5))

    # model.add(Conv1D(256, 5, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling1D())
    # model.add(Dropout(0.5))

    # # Flatten before dense layers if needed
    # model.add(Flatten())

    # # Example Dense layers
    # model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))

    # model.add(Normalization())


    # model.add(Embedding(input_dim=len(x_train), output_dim=256, input_length=sequence_length)) 


    # model.add(Conv1D(128, 7, padding='same', activation='relu'))   #od coveka sa gita
    # model.add(BatchNormalization())
    # model.add(MaxPooling1D(5))
    # model.add(Dropout(0.3))
    

    # model.add(Conv1D(256, 5, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling1D(5))
    # model.add(Dropout(0.3))

    # model.add(Conv1D(128, 5, padding='same', activation='relu'))
    # model.add(MaxPooling1D(35))  # Global max pooling
    # model.add(Dropout(0.3))

    # model.add(Flatten())
    # model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    # model.add(Dropout(0.3))
    # model.add(Dense(1, activation='sigmoid',kernel_regularizer=regularizers.l2(0.001))) 


   
    # opt = Adadelta(learning_rate=0.0001)
    # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # model.build(input_shape=(None, sequence_length))

    model.compile(loss='binary_crossentropy', # define the loss function
              optimizer=Adam(learning_rate=.001)) 
    #model.build(input_shape=(None, sequence_length))

    model.build(input_shape=(None, 1081))


    saving.save_model(model,"data/model/model.keras")

    # model_load = models.load_model("data/model/model.keras")
    # print(model_load)



if __name__ == '__main__':
    X_train, X_test, X_val, Y_train, Y_test, Y_val = split_dataset()
    print("X_train")
    print(len(X_train))
    create_model(X_train)
