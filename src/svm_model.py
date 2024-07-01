import pandas, re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.001, n_iters=90): 
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None


    def remove_stop_words(self):       
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')

        file_path = 'data/train.csv'
        text = pandas.read_csv(file_path, encoding='utf-8')

        text['labels'] = text['label']
        text['filtered_text'] = (text['textic']+text['title']+text['author']).apply(self.filter_text_column)

        # self.texts=text['filtered_text'].values
        # self.sentiments=text['labels'].values

        return text['filtered_text'].values,text['labels'].values


    def filter_text_column(self, text):
        if(not isinstance(text, str)):
            return ''
        
        text = re.sub(r'[^\w\s]', '', text)
        lemmatizer = WordNetLemmatizer()

        word_tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_words = [lemmatizer.lemmatize(w.lower()) for w in word_tokens if not w.lower() in stop_words]
        return ' '.join(filtered_words)


    def fit(self, X, y):
        print("dosaooo")
        nsamples, n_features = X.shape
        y_ = np.where(y == 0, -1, 1) 

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]


    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.where(np.sign(approx) == -1, 0, 1) 
    

if __name__=="__main__":

    vectorizer = TfidfVectorizer(max_features=2000)
    svm = SVM()
    text,sentiments = svm.remove_stop_words()
    X = vectorizer.fit_transform(text).toarray()
    y = sentiments

    # X, y = X[:1500], y[:1500]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123)

    svm.fit(x_train, y_train)

    y_pred = svm.predict(x_test)

    print()
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print()
    print(classification_report(y_test, y_pred, zero_division=0))
