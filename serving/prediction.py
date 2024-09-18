import pickle
import joblib
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from modul_cleaning import remove_stop_words, remove_digits, remove_punctuation
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score, ConfusionMatrixDisplay, make_scorer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

# Load environment variables
load_dotenv()

# Database connection details
db_host = os.getenv('DB_HOST_MODEL')
db_port = os.getenv('DB_PORT_MODEL')
db_user = os.getenv('DB_USER_MODEL')
db_password = os.getenv('DB_PASSWORD_MODEL')
db_name = os.getenv('DB_NAME_MODEL')

def load_data_from_db(query):
    """Load data from PostgreSQL database using a SQL query."""
    try:
        # Establish a connection to the database
        with psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        ) as conn:
            # Execute the query and load data into a DataFrame
            df = pd.read_sql_query(query, conn)
            print('Before Drop Duplicated', df.shape)
            
            # Drop duplicates
            df = df.drop_duplicates(keep='first')
            print('After Drop Duplicated', df.shape)
        
        return df
    except psycopg2.OperationalError as e:
        print(f"OperationalError connecting to the database: {e}")
    except Exception as e:
        print(f"Error connecting to the database: {e}")
    
    # Return an empty DataFrame in case of error
    return pd.DataFrame()

# Load data
query = 'SELECT * FROM "pms"."a6_02_risk_register_staging";'
df = load_data_from_db(query)

print(df.head(2))

unlabeled_data = df[df['CATEGORY FOR PREDICTION'].isnull()]

print(unlabeled_data.shape)

# Load model
with open('app/serving/model/best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Load data
with open('app/serving/model/data_preprocess.pkl', 'rb') as file:
    X_train_transformed, y_train = pickle.load(file)

# Load label encoder
with open('app/serving/model/label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

vectorizer = joblib.load('app/serving/model/vectorizer.pkl')

# Data input
data_input = unlabeled_data.head(1000)

def cleaning_data_input(data_input):
    '''Cleaning data input'''

    # Remove digits
    columns_to_clean = ['description', 'cause', 'impact_consequences']
    for col in columns_to_clean:
        data_input.loc[:, col] = data_input[col].map(remove_digits)
        
    # Remove punctuation
    for col in columns_to_clean:
        data_input.loc[:, col] = data_input[col].map(remove_punctuation)

    # Remove stop words
    data_input.loc[:, 'cause'] = data_input['cause'].map(lambda text: remove_stop_words(text) if text is not None else text)
    data_input.loc[:, 'description'] = data_input['description'].map(lambda text: remove_stop_words(text) if text is not None else text)
    data_input.loc[:, 'impact_consequences'] = data_input['impact_consequences'].map(lambda text: remove_stop_words(text) if text is not None else text)
    
    return data_input

data_cleaning = cleaning_data_input(data_input)

def is_test_value(text):
    '''Define invalid data'''
    if not isinstance(text, str):
        return False
    return text.strip().lower() == 'test'

def invalid_data(data_cleaning):
    '''Filter out rows with invalid data'''
    # Create a mask for rows with invalid values
    mask = data_cleaning[['description', 'cause', 'impact_consequences']].apply(lambda row: any(is_test_value(val) for val in row), axis=1)
    
    # Filter out rows that match the mask
    data_cleaning_filtered = data_cleaning[~mask].copy()
    
    return data_cleaning_filtered

# Use the function to filter the data_cleaning DataFrame
data_cleaning = invalid_data(data_cleaning)

def predict_data_input(data_input, vectorizer, best_model, label_encoder):
    '''Predict with the best model and inverse transform the predictions'''

    # Combine text data
    text_data = data_input[['description', 'cause', 'impact_consequences']].astype(str).agg(' '.join, axis=1)
    
    # Transform data using vectorizer
    X_transformed = vectorizer.transform(text_data)
    
    # Make predictions
    y_pred_numeric = best_model.predict(X_transformed)
    
    # Inverse transform to get original categories
    y_pred_original = label_encoder.inverse_transform(y_pred_numeric)
    
    # Add predictions and the probability of the predicted class to the DataFrame
    data_input['CATEGORY FOR PREDICTION'] = y_pred_original
    
    return data_input

predict = predict_data_input(data_input=data_cleaning, 
                             vectorizer=vectorizer,
                             label_encoder=label_encoder,
                             best_model=best_model)

# Print predictions for verification
print(predict)
print(predict['CATEGORY FOR PREDICTION'].unique())

import psycopg2

def update_db_with_predictions(predictions_df):
    """Update the PostgreSQL database with the predicted categories."""
    try:
        with psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        ) as conn:
            with conn.cursor() as cursor:
                for index, row in predictions_df.iterrows():
                    if pd.notna(row['CATEGORY FOR PREDICTION']):
                        query = """
                        UPDATE pms.a6_02_risk_register_staging
                        SET "CATEGORY FOR PREDICTION" = %s
                        WHERE "cut_off_date" = %s AND "risk_code" = %s;
                        """
                        params = (
                            row['CATEGORY FOR PREDICTION'],
                            row['cut_off_date'],
                            row['risk_code']
                        )
                        cursor.execute(query, params)
                
                conn.commit()
                print(f"Successfully updated {len(predictions_df)} rows in the database.")
    
    except psycopg2.OperationalError as e:
        print(f"OperationalError connecting to the database: {e}")
    except Exception as e:
        print(f"Error updating the database: {e}")

# Ensure that `predict` DataFrame has all the columns used in the WHERE clause
update_db_with_predictions(predict)
