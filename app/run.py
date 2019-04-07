import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/Messages.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/disaster_model.sav")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for graph 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # extract data needed for graph 2
    category_means = []
    for category in df.iloc[:,4:].columns:
        category_means.append(df[category].mean())

    df_means = pd.DataFrame(data={'categories': df.iloc[:,4:].columns,
    'means': category_means})
    df_means = df_means.sort_values(by=['means'], ascending=False)

    # extract data needed for graph 3
    df_cat = df.iloc[:,4:]
    df_cat['sum'] = df_cat.sum(axis=1)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=df_means['categories'],
                    y=df_means['means']
                )
            ],

            'layout': {
                'title': 'Proportion of Positive Labels',
                'yaxis': {
                    'title': "Proportion (%)"
                },
                'xaxis': {
                    'title': "",
                    'tickangle': -45
                }
            }
        }
        ,
        {
            'data': [
                Histogram(
                    x=df_cat['sum']
                )
            ],

            'layout': {
                'title': 'Histogram of Number of Categories per Message',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Number of Categories"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()
