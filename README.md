# Fake-News-Detection
This project utilizes machine learning algorithms and natural language processing (NLP) techniques to detect fake news articles. The goal is to classify news articles as real or fake based on their text content. 
It shows complete text preprocessing, feature extraction with TF-IDF, and model training using several ML algorithms, including Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.

# Features

Text cleaning, which includes removing punctuation, stopwords, and unwanted symbols

Tokenization and lemmatization with NLTK

TF-IDF vectorization for feature extraction

Training and evaluating multiple models

Performance metrics: accuracy, precision, recall, F1-score, and confusion matrix

Easily expandable for other NLP-based classification problems


# Dataset

The dataset includes labeled news articles with:

Text, representing the content of the article

Label, where 1 indicates fake news and 0 indicates real news

You can use any fake news dataset, such as the Fake and Real News Dataset.


# Text Preprocessing

The text data is cleaned and prepared by:

Removing punctuation and special characters using re and string

Lowercasing all words

Removing stopwords with nltk.corpus.stopwords

Lemmatizing words using WordNetLemmatizer


# Model Training

After preprocessing, TF-IDF extracts text features. Then, multiple classifiers are trained:

Logistic regression

Decision tree

Random forest

Gradient boosting


# Model Evaluation

Performance is measured by:

Accuracy

Precision

Recall

F1-score

Confusion matrix


# Results

Model                Accuracy    Precision    Recall    F1 Score  
Logistic Regression   ~0.95      ~0.94       ~0.95      ~0.94  
Decision Tree        ~0.90      ~0.89       ~0.90      ~0.89  
Random Forest        ~0.96      ~0.95       ~0.96      ~0.95  
Gradient Boosting    ~0.97      ~0.96       ~0.97      ~0.96  


# Future Improvements

Integrate deep learning models like LSTM or BERT for better contextual understanding

Deploy the model using Flask or FastAPI

# How to Run the Project

Clone the repository

git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection


Install dependencies

pip install -r requirements.txt


Download NLTK resources (only first time)

import nltk
nltk.download('stopwords')
nltk.download('wordnet')


Run the script

python main.py


# Learnings

Through this project, you’ll learn:

End-to-end text preprocessing for NLP

Managing imbalanced datasets
