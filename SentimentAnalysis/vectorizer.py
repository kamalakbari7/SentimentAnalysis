# SenimentAnalysis/vectorizer.py
from sklearn.feature_extraction.text import TfidfVectorizer

class TextVectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, text_data):
        return self.vectorizer.fit_transform(text_data)

    def transform(self, text_data):
        return self.vectorizer.transform(text_data)
