**Recognising Indic Languages using Wikipedia Data: An Experiment with Logistic Regression.**

This script experiments with a logistic regression model to differentiate between sentences from four different languages: Sanskrit, Hindi, Nepali, and English. It is based purely on data scraped from Wikipedia, which has articles in all four languages. Random articles from all four languages are downloaded using Python's *wikipedia* library, and the results are stored locally in a CSV file using Pandas.

All three Indic languages are written using Devanagari in the Wikipedia articles. The text is prepared by converting the Devanagari text data into Roman script using the *devtrans* module for Indic language text transliteration. The script converts the Devanagari text into the Harvard-Kyoto system, which uses only ASCII characters.

The script balances the data downloaded to ensure that each language has the same number of sentences. The balanced dataset is vectorised and used to train a logistic regression model.

When run, the model usually returns an accuracy score of 98%+.
