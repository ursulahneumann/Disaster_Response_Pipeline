import sys
import re
import nltk
import pickle
import pandas as pd

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def load_data(database_filepath):
    """Loads sqlite database into a dataframe then defines X and Y variables

    Args:
        database_filename: string - name of database (ex. 'Messages.db')

    Returns:
        X: dataframe - df containing features (the messages)
        Y: dataframe - df containing labels (the categories)
        category_names: column names of Y df (cateegory names)
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """Tokenizes text by performing: case normalization, punctuation removal,
    word tokenization, stop word removal, and stemming.

    Args:
        text: string - messages to be tokenized

    Returns:
        words: list of strings - tokenized words of text
    """
    # Case normalization
    text = text.lower()

    # Puncuation removal
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    # Tokenize words
    words = word_tokenize(text)

    # Stop word removal
    stop_words = stopwords.words("english")
    words = [w for w in words if w not in stop_words]

    # Perform stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words if w not in stop_words]

    return words


def build_model():
    """
    Performs machine learning pipeline

    Args:
        None

    Returns:
        model: trained model
    """
    # Build ML pipeline using random forest classifier
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(
        n_estimators=100, min_samples_split=2)))
    ])

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates model by calculating f1-score, precision, and recall

    Args:
        model: trained model
        X_test: list - train-test split of inputs
        Y_test: list - train-test split of inputs
        category_names: column names of Y df (cateegory names)

    Returns:
        report: dataframe - accuracy report
    """

    # Predict labels using model
    Y_pred = model.predict(X_test)

    # Generate accuracy report
    report = pd.DataFrame.from_dict(classification_report(Y_test, Y_pred,
        target_names=category_names, output_dict=True))
    report = pd.DataFrame.transpose(report)

    return report


def save_model(model, model_filepath):
    """
    Saves model as pickle file

    Args:
        model: trained model
        model_filepath: string - filepath of where model will be saved

    Returns:
        None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
