# Text-Classification-using-NLP 

This project performs sentiment analysis on IMDB movie reviews using various natural language processing techniques 
and machine learning models. It preprocesses the data, trains classifiers, evaluates their performance, and visualizes the results.

## Features

- **Data Preprocessing**: Cleans and preprocesses IMDB movie reviews using custom preprocessing steps.
- **Text Vectorization**: Implements TF-IDF and Count Vectorization methods for text representation.
- **Model Training**: Utilizes several classifiers including Logistic Regression, Random Forest, MLP Classifier, Multinomial Naive Bayes, Gaussian Naive Bayes, and Decision Tree Classifier.
- **Evaluation**: Computes accuracy metrics, generates classification reports, and visualizes results using confusion matrices, ROC curves, and prediction confidence analysis.
- **Visualization**: Provides visual insights into model performance and misclassified reviews.

## Installation and Setup

1. **Clone the Repository**:

   ```bash
   git https://github.com/ahmedsalim3/Text-Classification-using-NLP.git
   cd Text-Classification-using-NLP
   ```
2. **Download Dataset**:
   - Download the IMDB dataset from [Stanford AI Lab](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz).
   - Extract the contents and place the 'aclImdb' folder in the project directory.
  
3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
## Usage

Modify and run main.py to experiment with different preprocessing methods and models:

```bash
python main.py
```
### Parameters in main.py

- dataset_path: Path to the IMDB dataset directory (./aclImdb/train).
- sample_size: Number of samples to preprocess (200 in the example).
- test_size: Fraction of the dataset used for testing (0.15 in the example).
- methods: List of vectorization methods (['TfidfVectorizer', 'CountVectorizer']).
- models: List of classifiers (['LogisticRegression', 'RandomForestClassifier', 'MLPClassifier', 'MultinomialNB', 'GaussianNB', 'DecisionTreeClassifier']).

After running main.py, the best performing model based on accuracy will be displayed along with its performance metrics.
Feel free to explore the example results in [results.ipynb](https://github.com/Ahmeds-Data/Text-Classification-using-NLP/blob/master/results.ipynb)

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/Ahmeds-Data/Text-Classification-using-NLP/blob/master/LICENSE) file for more details.

## Credits

- [StemAway](https://stemaway.com/home)
- [Code Along for F-IE-1: Text Classification using NLP](https://stemaway.com/t/code-along-for-f-ie-1-text-classification-using-nlp/16302/1)

- [![Hits](https://hits.sh/github.com/Ahmeds-Data/Text-Classification-using-NLP.git.svg?label=views&color=cc11ac)](https://hits.sh/github.com/Ahmeds-Data/Text-Classification-using-NLP.git/)
