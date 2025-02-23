# SenimentAnalysis/evaluator.py
from sklearn.metrics import classification_report

class Evaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        report = classification_report(y_true, y_pred)
        print("Classification Report:\n", report)
