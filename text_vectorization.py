from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import nltk


def encode_labels(y):
    y_encoded = y.copy()
    y_encoded[y_encoded == 'positive'] = 1
    y_encoded[y_encoded == 'negative'] = 0
    return y_encoded.astype(int)


class TextVectorizer:
    def __init__(self, stopwords, max_features=5000):
        self.tokenizer = CountVectorizer().build_tokenizer()
        self.stop_words_tokens = self.tokenizer(" ".join(stopwords))
        self.count_vectorizer = CountVectorizer(max_df=0.8, min_df=3, tokenizer=nltk.word_tokenize,
                                                stop_words=stopwords, max_features=max_features)
        self.tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=3, tokenizer=nltk.word_tokenize,
                                                stop_words=self.stop_words_tokens, max_features=max_features)

    def fit_transform_count(self, X_train):
        countvect_train = pd.DataFrame(self.count_vectorizer.fit_transform(X_train).toarray(),
                                       columns=self.count_vectorizer.get_feature_names_out())
        return countvect_train

    def transform_count(self, x_test):
        countvect_test = pd.DataFrame(self.count_vectorizer.transform(x_test).toarray(),
                                      columns=self.count_vectorizer.get_feature_names_out())
        return countvect_test

    def fit_transform_tfidf(self, x_train):
        tfidfvect_train = pd.DataFrame(self.tfidf_vectorizer.fit_transform(x_train).toarray(),
                                       columns=self.tfidf_vectorizer.get_feature_names_out())
        return tfidfvect_train

    def transform_tfidf(self, x_test):
        tfidfvect_test = pd.DataFrame(self.tfidf_vectorizer.transform(x_test).toarray(),
                                      columns=self.tfidf_vectorizer.get_feature_names_out())
        return tfidfvect_test

# References
# sklearn CountVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# sklearn CountVectorizer: https://www.geeksforgeeks.org/using-countvectorizer-to-extracting-features-from-text/
# sklearn TfidfVectorizer: https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a
# sklearn TfidfVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
