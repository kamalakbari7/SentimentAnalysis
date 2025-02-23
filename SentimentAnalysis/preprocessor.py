# SenimentAnalysis/preprocessor.py
class Preprocessor:
    def __init__(self):
        pass

    def label_reviews(self, rating):
        if rating >= 4:
            return 1
        elif rating <= 2:
            return 0
        else:
            return 0

    def preprocess(self, df):
        df["label"] = df["rating"].apply(self.label_reviews)
        return df
