from nltk.corpus import stopwords
import nltk
import pandas
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer



class NaiveBayes:
  
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()
        self.texts=[]
        self.tests=[]
        self.sentiments=[]
        self.submit_sentiments=[]
        self.fake_word_counts={}
        self.true_word_counts={}
        self.text_counts = {'true': 0, 'fake': 0} 
        self.n_words = {'true': 0, 'fake': 0} 
        self.prior = {'true': 0, 'fake': 0}
        self.vocab_size = 0


    def remove_stop_words(self):
        # nltk.download('stopwords')
        # nltk.download('punkt')
        # nltk.download('wordnet')

        file_path = 'data/train.csv'
        text = pandas.read_csv(file_path, encoding='utf-8')

        file_path_test = 'data/test.csv'
        test = pandas.read_csv(file_path_test, encoding='utf-8')

        file_path_submit = 'data/submit.csv'
        submit = pandas.read_csv(file_path_submit, encoding='utf-8')


        text['labels'] = text['label']
        submit['labels']=submit['label']
        text['filtered_text'] = (text['textic']+text['title']+text['author']).apply(self.filter_text_column)
        test['filtered_test'] = (test['textic']+test['title']+test['author']).apply(self.filter_text_column)


        self.texts=text['filtered_text'].values
        self.sentiments=text['labels'].values
        self.tests=test['filtered_test'].values
        self.submit_sentiments=submit['labels'].values


    def filter_text_column(self,text):
        if(not isinstance(text, str)):
            return ''
        text = re.sub(r'[^\w\s]', '', text)
        word_tokens = word_tokenize(text)
        filtered_words = [self.lemmatizer.lemmatize(w.lower()) for w in word_tokens if not w.lower() in self.stop_words]
        return ' '.join(filtered_words)


    def fit(self) -> None:
        # X_train_tfidf = self.vectorizer.fit_transform(self.texts)
        # self.vocab_size = len(self.vectorizer.get_feature_names_out())
        # for text_vector, sentiment in zip(X_train_tfidf, self.sentiments):

        for text, sentiment in zip(self.texts, self.sentiments):
            # indices = text_vector.nonzero()[1]
            words = text.split()
            for word in words: 
            # for idx in indices:
                # word = self.vectorizer.get_feature_names_out()[idx]
                if str(sentiment) == '1': self.fake_word_counts[word] = self.fake_word_counts.get(word, 0) + 1
                if str(sentiment) == '0': self.true_word_counts[word] = self.true_word_counts.get(word, 0) + 1

        self.text_counts['fake'] = len([s for s in self.sentiments if str(s)=='1'])
        self.text_counts['true'] = len([s for s in self.sentiments if str(s)=='0'])

        self.n_words['fake'] = sum(self.fake_word_counts.values())
        self.n_words['true'] = sum(self.true_word_counts.values())
        
        n_total_texts = sum(self.text_counts.values())
        self.prior['fake'] = self.text_counts['fake'] / n_total_texts
        self.prior['true'] = self.text_counts['true'] / n_total_texts
    

    def predict(self, text: str) -> tuple[float, float]:
        words = text.split()
        
        log_p_words_given_fake_sentiment = []
        log_p_words_given_true_sentiment = []

        
        for word in words:
            p_word_given_fake = (self.fake_word_counts.get(word, 0) + 1) / (self.n_words['fake'] + len(self.fake_word_counts))
            p_word_given_true = (self.true_word_counts.get(word, 0) + 1) / (self.n_words['true'] + len(self.true_word_counts))

            log_p_words_given_fake_sentiment.append(np.log(p_word_given_fake))
            log_p_words_given_true_sentiment.append(np.log(p_word_given_true))


        log_p_text_given_fake = np.sum(log_p_words_given_fake_sentiment)
        log_p_text_given_true = np.sum(log_p_words_given_true_sentiment)

        log_p_text_is_fake = log_p_text_given_fake + np.log(self.prior['fake'])
        log_p_text_is_true = log_p_text_given_true + np.log(self.prior['true'])

        return log_p_text_is_fake>log_p_text_is_true


    def evaluate_model(self,true_labels, predicted_labels):
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        cm = confusion_matrix(true_labels, predicted_labels)
        
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-score: {f1:.2f}")
        print("Confusion Matrix:")
        print(cm)


if __name__=="__main__":
    bayes=NaiveBayes()
    bayes.remove_stop_words()
    bayes.fit()

    results=[]
    count=0

    for test in bayes.tests:
        is_fake=bayes.predict(test)
        if(is_fake):
            results.append(1)
        else:
            results.append(0)

    bayes.evaluate_model(bayes.submit_sentiments,results)
   

