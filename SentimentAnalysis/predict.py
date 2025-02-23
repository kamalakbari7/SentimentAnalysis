import argparse
import joblib

def load_model_and_vectorizer(model_path, vectorizer_path):
    """Load the trained model and vectorizer."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    """
    Predict sentiment for a given piece of text.
    """
    text_vector = vectorizer.transform([text])
    predicted_label = model.predict(text_vector)
    return "Positive" if predicted_label[0] == 1 else "Negative"

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Predict sentiment from a text input.")
    parser.add_argument("text", type=str, help="The text to classify.")
    args = parser.parse_args()

    # Load model and vectorizer (update paths as needed)
    model, vectorizer = load_model_and_vectorizer("SentimentAnalysis/trained_model.pkl", "SentimentAnalysis/vectorizer.pkl")

    # Make a prediction
    result = predict_sentiment(args.text, model, vectorizer)

    # Print the result
    print(f"Prediction: {result}")