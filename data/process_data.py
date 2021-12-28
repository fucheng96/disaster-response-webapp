# Import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """    
    INPUT:
        messages_filepath - File path to the messages in CSV format
        categories_filepath - File path to the message categories in CSV format
    Output:
        df - Combined dataset containing messages and its categories
    """
    
    # Import disaster messages and its categories
    msgs = pd.read_csv(messages_filepath)
    msgs_cat = pd.read_csv(categories_filepath)
    
    # Messages dataset to left-join its message categories
    df = msgs.merge(msgs_cat, how='left', on='id')
    
    return df 


def clean_data(df):
    
    """    
    INPUT:
        df - Combined dataset containing messages and its categories
    Output:
        df_cleaned - Cleansed dataset with indicator columns of each response category
    """
    
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # Select the first row of the categories dataframe as column names
    row = list(categories.loc[0])

    # Remove last 2 characters of each string
    category_colnames = [x[:-2] for x in row]

    # Rename the columns of `categories` dataset
    categories.columns = category_colnames

    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # Convert column from string to integer
        categories[column] = categories[column].astype(int)

    # Remove original original categories column from `df` dataset
    df.drop(columns='categories', inplace=True)

    # Concatenate the original dataframe with the new `categories` dataset
    df_cleaned = pd.concat([df, categories], axis=1)
    
    # Remove any rows having any category indicators of 2
    for column in category_colnames:
        df_cleaned = df_cleaned[(df_cleaned[column] >= 0) & (df_cleaned[column] <= 1)]

    # drop duplicates
    df_cleaned.drop_duplicates(inplace=True)
    
    return df_cleaned


def save_data(df, database_filename):
    
    """    
    INPUT:
        df - Combined dataset containing messages and its categories
        database_filename - Name of database which should not end with ".db"
    """
    
    # Create the SQL engine
    engine = create_engine('sqlite:///' + database_filename + '.db')
    
    # Output SQL database
    df.to_sql(database_filename, engine, index=False, if_exists='replace')


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