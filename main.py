from src import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def main(dataset_path, method, model, sample_size, test_size):
    preprocessor = TextPreprocessor(dataset_path, sample_size=sample_size)
    clean_df = preprocessor.preprocess()
    classifier = TextClassifier(clean_df)

    classifier.splitting(test_size=test_size)
    classifier.transform_data(method=method)
    trained_model = classifier.train_model(model=model)
    accuracy, report = classifier.evaluate_model(trained_model)

    visualizer = TextVisualizer(classifier.y_test, trained_model.predict(classifier.X_test_vect), method, model)
    visualizer.misclassified_reviews(classifier.X_test, trained_model, classifier.X_test_vect)
    visualizer.confusion_matrix()
    visualizer.roc_curve(trained_model, classifier.X_test_vect)
    visualizer.prediction_confidence_vs_length(trained_model, classifier.X_test_vect)

    return accuracy, model


if __name__ == "__main__":
    # Note: Before running this script, download the IMDB dataset from
    # http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    # Extract the contents and ensure you have a folder named 'aclImdb' in your working directory
    dataset_path = "./aclImdb/train"
    sample_size = 200
    test_size = .15
    methods = ['TfidfVectorizer', 'CountVectorizer']
    models = [
        'LogisticRegression', 'RandomForestClassifier',
        'MLPClassifier', 'MultinomialNB',
        'GaussianNB', 'DecisionTreeClassifier'
    ]

    best_acc, best_model = 0.0, None
    for method in methods:
        for model in models:
            acc, current_model = main(dataset_path, method, model, sample_size, test_size)
            if acc > best_acc:
                best_acc = acc
                best_model = current_model

    print(f"\n{'-' * 30}\nBest model based on accuracy: {best_model}, Accuracy: {best_acc}\n{'-' * 30}")

# # simple example usage
# if __name__ == "__main__":
#     dataset_path = "./aclImdb/train"
#     method = 'TfidfVectorizer' # or CountVectorizer
#     model = "RandomForestClassifier" # or 'MLPClassifier', 'LogisticRegression', 'MultinomialNB', 'GaussianNB', or 'DecisionTreeClassifier'
#     sample_size = 200
#     test_size = .15

#     main(dataset_path, method, model, sample_size, test_size)
