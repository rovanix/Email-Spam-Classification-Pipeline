# Email Spam Classification Pipeline

This project implements a complete machine learning pipeline for classifying emails as "Spam" or "Not Spam" (Ham).

## Overview
The project takes a dataset of raw emails, cleans and preprocesses the text, converts it into numerical features, and trains a Naive Bayes classifier to distinguish between spam and legitimate messages.

## Project Structure
The pipeline consists of the following key stages:

1.  **Data Loading & EDA**: Loads the dataset and explores class distribution (spam vs. ham).
2.  **Preprocessing**: Cleans raw text by lowercasing, removing punctuation/digits, and filtering out stopwords.
3.  **Feature Extraction**: Converts text to numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency).
4.  **Modeling**: Trains a Multinomial Naive Bayes classifier.
5.  **Evaluation**: Assessing model performance using accuracy, precision, recall, and a confusion matrix.
6.  **Deployment**: A custom function `classify_email` that takes raw text input and returns a prediction with a confidence score.

## Performance
The model achieves high performance on the test set:
* **Accuracy**: ~99%
* **Precision (Spam)**: 1.00 (Zero False Positives)
* **Recall (Spam)**: 0.97

## Requirements
To run this notebook, you will need the following Python libraries:

```text
pandas
numpy
matplotlib
seaborn
nltk
scikit-learn
wordcloud
