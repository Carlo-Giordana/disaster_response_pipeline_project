import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'omw-1.4'])

def load_data(database_filepath):
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM messages', engine)
    X = df.message.values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = Y.columns
    Y = Y.values
    
    return X, Y, category_names


def tokenize(text):
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator': [DecisionTreeClassifier(), KNeighborsClassifier()]
    }

    model = GridSearchCV(pipeline, param_grid=parameters, verbose=20)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)
    
    range_columns = range(len(Y_test[0])-1)
    range_rows_test = range(len(Y_test)-1)
    range_rows_pred = range(len(Y_pred)-1)
    
    report = dict.fromkeys(range_columns, [])
    for col in range_columns:
        Y_test_col = [Y_test[i][col] for i in range_rows_test]
        Y_pred_col = [Y_pred[i][col] for i in range_rows_pred]
        try:
            target_names = np.unique(Y_test_col).astype('str')
            report[col] = classification_report(Y_test_col, Y_pred_col, target_names=target_names)
        except:
            target_names = np.unique(Y_pred_col).astype('str')
            report[col] = classification_report(Y_test_col, Y_pred_col, target_names=target_names)
    for k in report.keys():
        print(report[k])


def save_model(model, model_filepath):
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