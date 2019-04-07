# Disaster Response Pipeline
This project makes use of data engineering to classify important tweets following a disaster.  The ETL pipeline processes message and category data from CSV and loads them into a SQLite database.  The machine learning pipeline reads in the database then creates and saves a multi-output supervised learning model. The web app extracts data from the database to create data visualizations and uses the model to classify a new message based on various categories.  

## Table of Contents
1) Project Motivation <br>
2) File Description <br>
3) Libraries Required <br>
4) Instructions <br>
5) Summary of Results <br>
6) Licensing and Acknowledgements <br>

## Project Motivation
To practice and showcase data engineering skills including using pipelines and a web app to work on the real world problem of classifying important messages following a disaster.

## File Descriptions
**README.md** - This file, describing the contents of this repo

**.gitignore** - The gitignore file

**ETL_Pipeline.ipynb** - Playground for the ETL pipeline .py file in the form of a Jupyter notebook

**ML_Pipeline.ipynb** - Playground for the ML pipeline .py file in the form of a Jupyter notebook

**app folder** - Contains the html files and the run.py file required to run the web app

**data folder** - Contains the categories and messages CSV files, the Mesages.db SQLite database containing cleaned message data, and the process_data.py file which is the summary of the ETL_Pipeline.ipynb file

**models folder** - Contains train_classifier.py file which is the summary of the ML_Pipeline.ipynb file


## Libraries Required
The code was written in python 3 and requires the following packages: re, sys, json, plotly, pandas, nltk, flask,
sklearn, sqlalchemy and pickle.  The following nltk data must be downloaded: punkt, stopwords, wordnet, and  averaged_perceptron_tagger.


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database: <br>
        `python data/process_data.py data/messages.csv data/categories.csv data/Messages.db`
    - To run ML pipeline that trains classifier and saves: <br>
        `python models/train_classifier.py data/Messages.db models/disaster_model.sav`

2. Run the following command in the app's directory to run your web app: <br>
    `python run.py`

3. Go to http://127.0.0.1:3001/

## Summary of Results
1) App landing page showing visualizations summarizing message and category data.
![app_visualizations](app_visualizations.png)

2) Example of a message classification.
![app_classification](app_classification.png)


## Licenses and Acknowledgements
This project was completed as part of the Udacity Data Scientist Nanodegree. Udacity provided the code templates and data. Figure Eight originally provided the data.
