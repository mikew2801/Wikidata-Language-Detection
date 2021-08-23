"""
Recognising Indic Languages using Logistic Regression and Wikipedia Data: An Experiment.

This script experiments with a logistic regression model to differentiate between sentences from four different languages: Sanskrit, Hindi, Nepali, and English. It is based purely on data scraped from Wikipedia, which has articles in all four languages. Random articles from all four languages are downloaded using Python's wikipedia library, and the results are stored locally in a CSV file using Pandas.

All three Indic languages are written using Devanagari in the Wikipedia articles. The text is prepared by converting the Devanagari text data into Roman script using the devtrans module for Indic language text transliteration. The script converts the Devanagari text into the Harvard-Kyoto system, which uses only ASCII characters.

The script balances the data downloaded to ensure that each language has the same number of sentences. The balanced dataset is vectorised and used to train a logistic regression model. When run, the model usually returns an accuracy score of 98%+.
"""

import wikipedia as wiki
import devtrans
import string
import re
import pandas as pd

# Configuration variables
# We'll only accept ascii letters, space marks, dandas (Indic punctuation marks) and full-stops
accepted_chars = string.ascii_letters + " " +  "।" + "."
num_articles = 1500
langs = ["sa", "hi", "en", "ne"]
min_sent_len = 20
max_sent_len = 200

# Functions to help prepare the data

def drop_rows_by_value(df, n_rows, col, value):
    """Drop a specified number of rows (n_rows) from from a pandas dataframe where the row's column contains the specified value."""
    selected = df[df[col] == value]
    selected = selected[0:n_rows]
    cut = df[df[col] != value]
    return pd.concat([selected, cut])

def balance_entries(df, col):
    """Ensures that the different languages have the same number of articles."""
    least = min(df[col].value_counts())
    for lang in langs:
        if df[col].value_counts()[lang] > least:
            df = drop_rows_by_value(df, least, "Tag", lang)
    return df

def process(text, lang):
    """Clean a single article by transliterating the text to Harvard-Kyoto format and removing everything apart from ASCII letters and basic punctuation."""
    sentences = []

    # If the language of the article is an Indic one, convert the Devanagari text into English using the devtrans library
    if lang in ("sa", "hi", "ne"):
        text = devtrans.dev2hk(text).replace("ळ", "")

    # Remove all characters apart from ASCII letters and basic punctuation
    cleaned = "".join([char for char in text if char in accepted_chars])

    # Split text into sentences according to punctuation markers
    sentences += [sentence for sentence in re.split("।|\.", cleaned) if len(sentence) > min_sent_len and len(sentence) < max_sent_len]
    return sentences

def download_articles(lang, how_many):
    """Downloads a specified number of articles on random topics from wikipedia in the language specified."""
    wiki.set_lang(lang)
    articles = []
    tags = []
    topics = wiki.random(how_many)
    for topic in topics:        
        try:
            sentences = process(wiki.WikipediaPage(topic).content, lang)
            for sentence in sentences:
                tags.append(lang)
            articles += sentences
        # In case the topic is really available on wikipedia
        except (wiki.PageError, wiki.DisambiguationError):
            print("Unable to retrieve content for: " + topic)
    return (articles, tags)

final_sentences = []
final_tags = []
sents = []
tags = []

# Download the data

for lang in langs:
    final_sentences.append(download_articles(lang, num_articles))

for group in final_sentences:
    for i, sent in enumerate(group[0]):
        sents.append(sent)
        tags.append(group[1][i])

# Convert the downloaded data into a dictionary with tags showing the language to train the model
zipped = {"Sentence":sents, "Tag":tags}
df = balance_entries(pd.DataFrame(zipped), "Tag")

# Write the data downloaded into a locally stored CSV file for future use
df.to_csv("sentences.csv")

# Training the model
X, y = df.iloc[:, 0], df.iloc[:,1]

from sklearn.model_selection import train_test_split
from sklearn import feature_extraction
from sklearn import pipeline
from sklearn import linear_model
from sklearn import metrics

# Split the data into independent/dependent variables and training/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# Vectorise the data for analysis
vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=(1,3), analyzer="char")

pipe = pipeline.Pipeline([("vectorizer", vectorizer),
                                 ("clf", linear_model.LogisticRegression())])

# Train the model and predict/score the test data
pipe.fit(X_train, y_train)
y_predicted = pipe.predict(X_test)
acc = (metrics.accuracy_score(y_test, y_predicted)) * 100
print("Accuracy = " + str(acc))

# Run some tests on strings from the different languages inputted in Harvard-Kyoto format

new_test = ["This is a string", "mujhe mAluM hai ki tumhArI", "lakSyatAvacchedakAvacchinnaM na bhavati", "pramAnalakSaNaM syAt", "My name is something or other", "These things are all nice", "pramANaprameyasaMzayaprayojanadRSTAntAdi", "Dave went to sleep at hai hai hai merA", "pratiyogitA kA nAma kyA zuklatvam iti hai"]

new_pred = pipe.predict(new_test)
print(new_pred)
