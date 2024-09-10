import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc


class TextVisualizer:
    def __init__(self, y_test, predictions, method, model):
        self.y_test = y_test
        self.predictions = predictions
        self.method = method
        self.model = model

    def confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.predictions)
        true_positives = np.diag(cm)
        total_actual = np.sum(cm, axis=1)
        cm_percent = np.zeros_like(cm, dtype=float)

        for i in range(len(cm)):
            cm_percent[i, i] = true_positives[i] / total_actual[i]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap="BuPu", fmt='d', linecolor='black', linewidths=.7)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i == j:
                    plt.text(j + 0.5, i + 0.8, f'{cm_percent[i, j] * 100:.2f}%', ha='center', va='center',
                             color='white')

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix\nModel: {self.model}\nMethod: {self.method}')
        plt.show()

    def roc_curve(self, clf, x_test_vect):
        y_score = clf.predict_proba(x_test_vect)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve\nModel: {self.model}\nMethod: {self.method}')
        plt.legend(loc='lower right')
        plt.show()

    def prediction_confidence_vs_length(self, clf, x_test_vect):
        predictions = clf.predict(x_test_vect)
        probabilities = clf.predict_proba(x_test_vect)
        confidence = probabilities.max(axis=1)
        review_lengths = np.arange(len(x_test_vect))

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=review_lengths, y=confidence, hue=predictions, palette='coolwarm', alpha=0.7)
        plt.title(f'Prediction Confidence vs Review Length\nModel: {self.model}\nMethod: {self.method}')
        plt.xlabel('Review Length')
        plt.ylabel('Prediction Confidence')
        plt.show()

    def misclassified_reviews(self, x_test, clf, x_test_vect):
        predictions = clf.predict(x_test_vect)
        misclassified_indices = np.where(predictions != self.y_test)[0]

        if len(misclassified_indices) == 0:
            print("No misclassified reviews found.")
            return

        print(f'Misclassified Reviews:\n')
        num_plots = min(len(misclassified_indices), 10)

        for idx in range(num_plots):
            mis_idx = misclassified_indices[idx]
            if mis_idx < len(x_test):
                try:
                    review_snippet = x_test.iloc[mis_idx][:60] + '...' if len(x_test.iloc[mis_idx]) > 100 else \
                        x_test.iloc[mis_idx]
                    print(f"{idx + 1}. {review_snippet}")

                except KeyError as e:
                    print(f"KeyError: {e} occurred at index {mis_idx}.")
                    continue
