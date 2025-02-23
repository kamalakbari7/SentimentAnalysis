
# Sentiment Analysis

## Overview

This project uses a machine learning pipeline to classify text sentiment. The main goal is to determine whether a given piece of text expresses a positive or negative sentiment. The workflow includes data loading, preprocessing, vectorization, model training, evaluation, and prediction.

## Project Structure

```plaintext
SentimentAnalysis/
│
├── Data/
│   └── data.csv                  # Input dataset (text and labels)
├── SentimentAnalysis/
│   ├── __init__.py
│   ├── data_loader.py            # Handles loading data from CSV files
│   ├── preprocessor.py           # Performs text preprocessing and labeling
│   ├── vectorizer.py             # Converts text into numerical features (TF-IDF)
│   ├── model.py                  # Defines and trains the machine learning model
│   ├── evaluator.py              # Evaluates model performance on test data
│   └── predict.py                # Makes predictions on new text inputs
├── trained_model.pkl             # Serialized trained machine learning model
├── vectorizer.pkl                # Serialized TF-IDF vectorizer
└── README.md                     # Project documentation 
```

## Prerequisites

-   **Python 3.8+**
-   **scikit-learn**
-   **pandas**
-   **joblib**

You can install the required Python packages using pip:

`pip install -r requirements.txt` 

## Features

### Data Loader

-   Reads raw text data and corresponding labels from a CSV file.

### Text Preprocessing

-   Tokenizes text
-   Normalizes case
-   Removes stop words
-   Assigns labels based on a rating threshold

### TF-IDF Vectorization

-   Converts cleaned text into numerical features suitable for machine learning.

### Model Training

-   Trains a logistic regression model to classify sentiment as positive or negative.

### Evaluation

-   Measures model performance using accuracy, precision, recall, and F1-score.

### Prediction

-   Provides a command-line interface to classify a single input text.

## Usage

### Training the Model

To train the model, ensure that your dataset is placed in the `Data/` directory as `data.csv`. Then run the main script:

`python main.py` 

This script will:

1.  Load the dataset.
2.  Preprocess and vectorize the text.
3.  Train a logistic regression model.
4.  Save the trained model and vectorizer for future use.

### Making Predictions

To predict the sentiment of a text, use the `predict.py` script:

`python -m SentimentAnalysis.predict "Your input text here"` 

The script will output whether the input is classified as positive or negative.

## Contributing

Contributions are welcome! If you’d like to suggest improvements or add new features, please fork the repository and submit a pull request.