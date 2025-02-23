# main.py
from SentimentAnalysis.data_loader import DataLoader
from SentimentAnalysis.preprocessor import Preprocessor
from SentimentAnalysis.vectorizer import TextVectorizer
from SentimentAnalysis.model import SentimentModel
from SentimentAnalysis.evaluator import Evaluator
import joblib

if __name__ == "__main__":
    # Example usage:
    # Load data
    # In a real scenario, replace 'data/reviews.csv' with the actual path
    # Here we're assuming reviews.csv contains 'review_text' and 'rating' columns
    loader = DataLoader("Data/data.csv")
    df = loader.load_data()

    # Preprocess data
    preprocessor = Preprocessor()
    df = preprocessor.preprocess(df)

    # Vectorize text
    vectorizer = TextVectorizer()
    X = vectorizer.fit_transform(df["review_text"])
    y = df["label"]

    # Split data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = SentimentModel()
    model.train(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    evaluator = Evaluator()
    evaluator.evaluate(y_test, y_pred)

   

    # Save the trained model
    joblib.dump(model, "SentimentAnalysis/trained_model.pkl")

    # Save the vectorizer
    joblib.dump(vectorizer, "SentimentAnalysis/vectorizer.pkl")

