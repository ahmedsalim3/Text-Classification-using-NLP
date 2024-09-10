import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import load_files
import pandas as pd

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


class TextPreprocessor:
    def __init__(self, dataset_path, sample_size=200):
        self.dataset = load_files(dataset_path, categories=['pos', 'neg'], shuffle=True, random_state=42)
        self.df = pd.DataFrame({'review': self.dataset.data, 'sentiment': self.dataset.target})
        self.df['review'] = self.df['review'].apply(lambda x: x.decode('utf-8'))
        self.df = self.df.sample(n=sample_size, random_state=42)
        self.replacement_patterns = [
            (r'won\'t', 'will not'),
            (r'can\'t', 'cannot'),
            (r'i\'m', 'i am'),
            (r'ain\'t', 'is not'),
            (r'(\w+)\'ll', '\g<1> will'),
            (r'(\w+)n\'t', '\g<1> not'),
            (r'(\w+)\'ve', '\g<1> have'),
            (r'(\w+)\'s', '\g<1> is'),
            (r'(\w+)\'re', '\g<1> are'),
            (r'(\w+)\'d', '\g<1> would')
        ]
        self.stopwords = stopwords.words('english')

    def search_review(self, target_str, pattern):
        cpat = re.compile(pattern)
        return bool(cpat.search(target_str))

    # For debugging
    # def check_if_contain(self, data, x, pattern, contain=True):
    #     temp_df = data.copy()
    #     temp_df[f'contains_{x}'] = temp_df.apply(lambda x: self.search_review(x['review'], pattern), axis=1)
    #     return temp_df[temp_df[f'contains_{x}'] == contain]

    def clean_html(self):
        self.df['review'] = self.df['review'].apply(lambda string: BeautifulSoup(string, 'html.parser').get_text())

    def clean_url(self):
        self.df['review'] = self.df['review'].str.replace(r"https://\S+|www\.\S+", '', regex=True)

    def standardize_casing(self):
        self.df['review'] = self.df['review'].apply(lambda x: x.lower())

    def replace_contractions(self):
        replacer = RegexpReplacer(self.replacement_patterns)
        self.df['review'] = self.df['review'].apply(lambda x: replacer.replace(x))

    def remove_stopwords(self):
        self.df['review'] = self.df['review'].apply(self._remove_stopwords)

    def _remove_stopwords(self, s):
        words = word_tokenize(s)
        filtered_words = [w for w in words if w not in self.stopwords]
        return " ".join(filtered_words)

    def clean_symbols(self):
        symbol_regex = r'[^a-z|\s]'
        self.df['review'] = self.df['review'].str.replace(symbol_regex, ' ', regex=True)

    def lemmatize_reviews(self):
        self.df['lemma_review'] = self.df['review'].apply(self._lemmatize_string)

    @staticmethod
    def _lemmatize_string(s):
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(s)
        lemmas = [lemmatizer.lemmatize(w) for w in words]
        return " ".join(lemmas)

    def preprocess(self):
        self.clean_html()
        self.clean_url()
        self.standardize_casing()
        self.replace_contractions()
        self.remove_stopwords()
        self.clean_symbols()
        self.lemmatize_reviews()

        return self.df


class RegexpReplacer:
    def __init__(self, patterns):
        self.patterns = [(re.compile(regex), repl) for regex, repl in patterns]

    def replace(self, text):
        s = text
        for pattern, repl in self.patterns:
            s = re.sub(pattern, repl, s)
        return s

# References
# Fix contractions: https://gist.github.com/yamanahlawat/4443c6e9e65e74829dbb6b47dd81764a "actually not required as they will be removed along with stopwords"
# remove stopwords: https://stackabuse.com/removing-stop-words-from-strings-in-python/#usingpythonsgensimlibrary
