import json
from nltk.corpus import stopwords
import nltk
import pandas
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter
import numpy
from keras.src.optimizers import Adam,Adadelta
from string import punctuation, digits
from keras import models,saving
from flatbuffers.builder import np
from nltk import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from keras.src.optimizers import Adam
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.callbacks import EarlyStopping,ReduceLROnPlateau




def remove_stop_words():

        file_path = 'data/train.csv'
        text = pandas.read_csv(file_path, encoding='utf-8')

        text['labels'] = text['label']
        text['filtered_text'] = (text['textic']+text['title']+text['author']).apply(filter_text_column)

        return text['filtered_text'].values,text['labels'].values


def filter_text_column(text):
        stop_words = set(stopwords.words('english'))
        if(not isinstance(text, str)):
            return ''
        text = re.sub(r'[^\w\s]', '', text)
        word_tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()   
        filtered_words = [lemmatizer.lemmatize(w.lower()) for w in word_tokens if not w.lower() in stop_words]
        return filtered_words

def preparing_data(text):
    max_features = 20000    
    sequence_lengths = [len(seq) for seq in text]
  
    max_token = int(np.percentile(sequence_lengths, 95))   

    # max_token = len(max(text, key=len))    #kada bi uzeli max preveliko je jer ima neki predugacak tekst
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    X = pad_sequences(sequences, maxlen=max_token)
    return X

def split_dataset():
    texts, is_fake = remove_stop_words()
    texts = preparing_data(texts)

    Y = np.vstack(is_fake)

    X_train, X_temp, Y_train, Y_temp = train_test_split(texts, Y, test_size=0.30, shuffle=True, random_state=123)   #70 15 15 preporuceno
    
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.50, shuffle=True, random_state=123)

    return  X_train, X_test, X_val, Y_train, Y_test, Y_val

def fit_model(model, X_train, Y_train, X_val, Y_val):

    print("Fit")

    y_ints = [int(y[0]) for y in Y_train]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_ints), y=y_ints)
    class_weight_dict = dict(enumerate(class_weights))

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)


    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=20, batch_size=256, verbose=2, class_weight=class_weight_dict,callbacks=[early_stopping])
    model.summary()
    model.save_weights("data/weights.weights.h5")
    

if __name__ == '__main__':

    # texts, is_fake = remove_stop_words()
    # texts = preparing_data(texts)

    X_train, X_test, X_val, Y_train, Y_test, Y_val = split_dataset()

    model = models.load_model("data/model/model.keras")
    opt = Adam(learning_rate=0.001) 
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    fit_model(model, X_train, Y_train, X_val, Y_val)