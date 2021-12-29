# Import core libraries
import sys
import re
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

# Import natural language toolkit libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

# Scikit learn libraries
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report


def load_data(database_filepath):
    
    """    
    INPUT:
        database_filepath - File path to SQLite database which should end with ".db"
    OUTPUT:
        X - Dataset containing all the model features
        Y - Dataset containing all 36 indicator columns for each response category
        category_colnames - List of categories name
    """
    
    # Create SQL engine to import SQLite database
    engine = create_engine('sqlite:///' + database_filepath)
    conn = engine.raw_connection()
    cur = conn.cursor()
    
    # Import data table from database
    database_filename = database_filepath.split('/')[1]
    database_filename = database_filename.replace('.db', '')
    sql_command = "SELECT * FROM " + database_filename
    df = pd.read_sql(sql_command, con=conn)
    conn.commit()
    conn.close()
    
    # Split dataset 'df' to features and target columns
    n_response_cols = 36 # Response variables are the last 36 columns
    X = df['message']
    y = df.iloc[:, -n_response_cols:]
    
    # Get response variables column names
    category_colnames = y.columns.values
    
    return X, y, category_colnames


def tokenize(text):
    
    """
    INPUT:
        text - Text message that would need to be tokenized
    OUTPUT:
        clean_tokens - List of tokens extracted from the text message
    """
    
    # Detect & remove punctuations from the message
    detected_punctuations = re.findall('[^a-zA-Z0-9]', text)   
    for punctuation in detected_punctuations:
        text = text.replace(punctuation, " ")

    # Tokenize the words
    tokens = word_tokenize(text)
    
    # Lemmanitizer to reduce words to its stems
    lemmatizer = WordNetLemmatizer()

    # Return list of normalized tokens reduced to its stems
    cleaned_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    return cleaned_tokens


def build_model():
    
    """
    OUTPUT:
        Machine Learning pipeline using AdaBoost to process and classify text messages
    """
    
    # Machine Learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    INPUT:
        model - Trained Machine Learning pipeline
        X_test - Test features
        Y_test - Test labels
        category_names - label names (multi-output)
    OUTPUT:
        Metrics including overall accuracy, precision, recall, f1-score & support
    """
    
    # Predict on test data using trained model
    Y_pred = model.predict(X_test)

    # Display metrics
    print("Overall accuracy: " + str(np.round(100 * (Y_pred == Y_test).mean().mean(),2)) + "%")
    print(classification_report(Y_test.values, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    
    """
    INPUT:
        model - Trained Machine Learning pipeline
        model_filepath - File path to where the model is saved
    OUTPUT:
        Model saved in .pkl format
    """
    
    # Save model
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