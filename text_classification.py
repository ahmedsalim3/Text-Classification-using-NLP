from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from text_vectorization import TextVectorizer, encode_labels


class TextClassifier:
    def __init__(self, cleaned_df, max_features=5000):
        self.cleaned_df = cleaned_df
        self.vectorizer = TextVectorizer(stopwords.words('english'), max_features=max_features)
        self.LogisticRegression = LogisticRegression()
        self.MLPClassifier = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                           hidden_layer_sizes=(5, 2), random_state=1, max_iter=200)
        self.MultinomialNB = MultinomialNB()
        self.GaussianNB = GaussianNB()
        self.RandomForestClassifier = RandomForestClassifier(n_estimators=1000, random_state=0)
        self.DecisionTreeClassifier = DecisionTreeClassifier(random_state=42)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.method = None
        self.X_test_vect = None
        self.X_train_vect = None

    def splitting(self, test_size=0.1):
        self.X = self.cleaned_df['lemma_review']
        self.y = self.cleaned_df['sentiment']
        self.y = encode_labels(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                random_state=42)

    def transform_data(self, method='CountVectorizer'):
        self.method = method
        if method == 'CountVectorizer':
            self.X_train_vect = self.vectorizer.fit_transform_count(self.X_train)
            self.X_test_vect = self.vectorizer.transform_count(self.X_test)
        elif method == 'TfidfVectorizer':
            self.X_train_vect = self.vectorizer.fit_transform_tfidf(self.X_train)
            self.X_test_vect = self.vectorizer.transform_tfidf(self.X_test)
        else:
            raise ValueError("Method must be 'CountVectorizer' or 'TfidfVectorizer'")

    def train_model(self, model='MLPClassifier'):
        if model == 'MLPClassifier':
            clf = self.MLPClassifier
        elif model == 'LogisticRegression':
            clf = self.LogisticRegression
        elif model == 'MultinomialNB':
            clf = self.MultinomialNB
        elif model == 'GaussianNB':
            clf = self.GaussianNB
        elif model == 'RandomForestClassifier':
            clf = self.RandomForestClassifier
        elif model == 'DecisionTreeClassifier':
            clf = self.DecisionTreeClassifier
        else:
            raise ValueError(
                "Model must be 'MLPClassifier','LogisticRegression', 'MultinomialNB', 'GaussianNB', 'RandomForestClassifier', or 'DecisionTreeClassifier'")

        clf.fit(self.X_train_vect, self.y_train)
        print(f"{'-' * 30}\n Model: {model}\n Method: {self.method}\n{'*' * 30}")
        return clf

    def evaluate_model(self, clf):
        accuracy = clf.score(self.X_test_vect, self.y_test)
        print(f'Accuracy: {accuracy:.2f}')

        predictions = clf.predict(self.X_test_vect)

        print('Classification Report:\n')
        print(classification_report(self.y_test, predictions, target_names=['negative', 'positive'], digits=2))
        report = classification_report(self.y_test, predictions, target_names=['negative', 'positive'], digits=2,
                                       output_dict=True)

        # avg_precision = report['weighted avg']['precision']
        # avg_recall = report['weighted avg']['recall']
        # avg_f1_score = report['weighted avg']['f1-score']

        # print(f'Average Precision: {avg_precision:.4f}')
        # print(f'Average Recall: {avg_recall:.4f}')
        # print(f'Average F1 Score: {avg_f1_score:.4f}')

        return accuracy, report
