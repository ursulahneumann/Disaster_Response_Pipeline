import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loads then merges messages and categories datasets

    Args:
        messages_filepath: string - filepath to messages.csv file
        categories_filepath: string - filepath to categories.csv file

    Returns:
        df: dataframe - merged df containing messages and categories data

    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # Merge datasets
    df = messages.merge(categories, on = 'id')

    return df

def clean_data(df):
    """Clean dataframe by slitting categories into their own columns with binary
    value, remove rows where 'related' = 2, remove the child_alone column, drop
    duplicates, and drop rows where no categorical info.

    Args:
        df: dataframe - merged df containing messages and categories data

    Returns:
        df: dataframe - cleaned df

    """

    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)

    # Select the first row of the categories dataframe
    row = categories.iloc[0]

    # Use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])

    # Rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:

        # Set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Remove rows where 'related' = 2
    categories = categories[categories['related'] != 2]

    # Remove 'child_alone' column which only has one label
    categories = categories.drop(['child_alone'], axis=1)

    # Drop the original categories column from 'df'
    df.drop(['categories'], axis=1, inplace=True)

    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Remove rows where categories are NaNs
    df = df.dropna(subset=['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report'])

     return df

def save_data(df, database_filename):
    """Saves the clean dataset into an sqlite database

    Args:
        df: dataframe - cleaned dataset
        database_filename: string - name of database (ex. 'Messages.db')

    Returns: None
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
