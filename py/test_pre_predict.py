import pytest
import pandas as pd
import pickle
import os
import sys
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, make_scorer
import string
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
import psycopg2
from psycopg2 import OperationalError
from dotenv import load_dotenv
import pytest
from unittest.mock import patch, Mock

# Menambahkan direktori 'serving' ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'serving')))

from modul_cleaning import remove_digits, remove_punctuation

def get_data_train():
    '''Load data X and y train'''

    # Nama direktori
    directory = r'app\data'

    # Nama file
    X_train_file = os.path.join(directory, 'X_train.pkl')
    y_train_file = os.path.join(directory, 'y_train.pkl')

    # Muat data dari file joblib
    X_train = joblib.load(X_train_file)
    y_train = joblib.load(y_train_file)
    
    # Debugging output
    print("Tipe X_train:", type(X_train))
    print("Tipe y_train:", type(y_train))
    if isinstance(y_train, (pd.DataFrame, pd.Series)):
        print("Beberapa baris y_train:\n", y_train.head())
    else:
        print("Konten y_train:\n", y_train)
    
    # Konversi y_train ke pandas Series jika perlu
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)

    elif isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]  

    elif not isinstance(y_train, pd.Series):
        raise TypeError("y_train harus berupa pandas DataFrame, Series, atau numpy ndarray")
    
    return X_train, y_train

def test_data_types():
    '''Validation column and type name in predictor'''
    
    # load data
    X_train, _ = get_data_train()

    # Define the expected columns and their data types
    expected_columns = ['description', 'cause', 'impact_consequences']
    expected_dtypes = {
        'description': 'object',
        'cause': 'object',
        'impact_consequences': 'object'
    }
    
    # Check if the columns in the DataFrame match the expected columns
    actual_columns = X_train.columns.tolist()
    if sorted(actual_columns) != sorted(expected_columns):
        raise ValueError(f"Unexpected columns. Expected: {expected_columns}, Found: {actual_columns}")

    # Check if the data types match the expected data types
    actual_dtypes = X_train.dtypes.to_dict()
    for column, expected_dtype in expected_dtypes.items():
        actual_dtype = actual_dtypes.get(column, None)
        if actual_dtype != expected_dtype:
            raise TypeError(f"Column '{column}' type mismatch. Expected: {expected_dtype}, Found: {actual_dtype}")

    print("Data types and columns are as expected.")

def test_label():
    '''Validation column and type name in target'''

    _, y_train = get_data_train()
    
    # Cetak tipe dan beberapa baris awal y_train untuk debugging
    print("Tipe y_train:", type(y_train))

    if isinstance(y_train, (pd.DataFrame, pd.Series)):
        print("Beberapa baris awal y_train:\n", y_train.head())

    elif isinstance(y_train, np.ndarray):
        print("Konten y_train:\n", y_train[:10])  
    
    # Konversi y_train ke pandas Series 
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)

    elif isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]  
    
    # Ambil label unik dari y_train
    actual_labels = sorted(set(y_train))
    print("Label yang ditemukan:", actual_labels)
    
    # Label yang diharapkan
    expected_labels = sorted(['contract terms', 'other', 'construction', 'scope', 'procurement', 'technical'])
    
    # Periksa apakah label yang sebenarnya cocok dengan label yang diharapkan
    assert actual_labels == expected_labels, f"Label yang diharapkan: {expected_labels}, tapi yang didapat: {actual_labels}"

def test_cleaning():
    '''Test for digits and punctuation in specified columns after cleaning'''
    
    # Load data
    X_train, _ = get_data_train()
    
    # Columns to check
    columns_to_check = ['description', 'cause', 'impact_consequences']
    
    # Apply cleaning functions
    for col in columns_to_check:
        X_train[col] = X_train[col].astype(str)
        X_train[col] = X_train[col].apply(remove_punctuation)  # Apply remove_punctuation function
        X_train[col] = X_train[col].apply(remove_digits)  # Apply remove_digits function

    # Check for digits using updated regex pattern
    has_digits = X_train[columns_to_check].astype(str).apply(lambda x: x.str.contains(r'\d', regex=True).any()).any()
    assert not has_digits, "One or more columns contain digits after cleaning."
    
    # Check for punctuation
    def contains_punctuation(s):
        """Check if a string contains punctuation."""
        return any(char in string.punctuation for char in s)
    
    # Apply check for punctuation across specified columns
    has_punctuation = X_train[columns_to_check].astype(str).apply(lambda x: x.apply(contains_punctuation).any()).any()
    assert not has_punctuation, "One or more columns contain punctuation after cleaning."

def get_model():
    '''Load model'''

    model_path = r'app\model\models\best_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # load model
    model = pd.read_pickle(model_path)
    
    # Load data
    data_preprocess_path = 'C:/Project/Data Science/NLP Project/serving/model/data_preprocess.pkl'
    if not os.path.exists(data_preprocess_path):
        raise FileNotFoundError(f"Data preprocess file not found: {data_preprocess_path}")

    with open(data_preprocess_path, 'rb') as file:
        X_train_transformed, y_train = pickle.load(file)

    return model, y_train, X_train_transformed

def test_metric():
    '''Test metric'''
    
    # load model
    model, y_train, X_train_transformed = get_model()

    # Fit and predict
    model.fit(X_train_transformed, y_train)
    y_pred = model.predict(X_train_transformed)
    
    # Compute macro F1 score
    macro_f1 = f1_score(y_train, y_pred, average='macro')

    # Transform data if needed
    X_train_transformed = X_train_transformed.toarray() if hasattr(X_train_transformed, 'toarray') else X_train_transformed

    # Handle ROC AUC Calculation
    try:
        if len(np.unique(y_train)) == 2:  # Binary classification
            roc_auc = roc_auc_score(y_train, model.predict_proba(X_train_transformed)[:, 1])
        else:  # Multiclass classification
            roc_auc = roc_auc_score(y_train, model.predict_proba(X_train_transformed), multi_class='ovr')
    except ValueError:
        roc_auc = np.nan  # Handle cases where ROC AUC cannot be computed
    
    # Calculate Balanced Accuracy
    balanced_acc = balanced_accuracy_score(y_train, y_pred)

    # Define threshold (convert to decimal if needed)
    threshold = 0.75

    # Validate metrics
    metrics_failed = False
    message = ""

    if np.isnan(roc_auc):
        metrics_failed = True
        message += "ROC AUC could not be computed. "

    if macro_f1 < threshold:
        metrics_failed = True
        message += f"Macro F1 score {macro_f1:.2f} is below the threshold of {threshold}. "

    if balanced_acc < threshold:
        metrics_failed = True
        message += f"Balanced Accuracy {balanced_acc:.2f} is below the threshold of {threshold}. "

    if metrics_failed:
        pytest.fail(message)

def test_cross_validation():
    '''Consistency of model across different subsets of data.'''
    model, y_train, X_train_transformed = get_model()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = {
        'f1_macro': 'f1_macro',
        'balanced_accuracy': make_scorer(balanced_accuracy_score)
    }
    
    cv_results = cross_validate(
        estimator=model,
        X=X_train_transformed,
        y=y_train,
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )
    
    # Extract F1 macro scores
    f1_scores = cv_results['test_f1_macro']
    
    # Check the consistency of the F1 scores
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    print(f"Mean F1 Score: {mean_f1:.4f}")
    print(f"Standard Deviation of F1 Scores: {std_f1:.4f}")
    
    # Define a threshold for acceptable variance
    threshold = 0.05 
    
    # Ensure the standard deviation is within the acceptable threshold
    assert std_f1 < threshold, f"F1 scores are inconsistent. Std deviation: {std_f1:.4f}"


def connect_db():
    # Memuat variabel lingkungan dari file .env
    load_dotenv()
    
    try:
        # Konfigurasi koneksi database
        connection = psycopg2.connect(
            host=os.getenv('DB_HOST_MODEL'),
            port=os.getenv('DB_PORT_MODEL'),
            user=os.getenv('DB_USER_MODEL'),
            password=os.getenv('DB_PASSWORD_MODEL'),
            dbname=os.getenv('DB_NAME_MODEL')
        )
        return connection
    except OperationalError as e:
        print(f"Error connecting to the database: {e}")
        return None

# Pengujian untuk fungsi connect_db
@patch('psycopg2.connect')  # Mock psycopg2.connect
def test_connect_db(mock_connect):
    # Konfigurasi mock untuk mengembalikan objek mock
    mock_connect.return_value = Mock()

    # Panggil fungsi yang ingin diuji
    connection = connect_db()

    # Periksa apakah koneksi berhasil
    assert connection is not None, "Database connection should be established"
    
    # Periksa apakah psycopg2.connect dipanggil dengan parameter yang benar
    mock_connect.assert_called_once_with(
        host=os.getenv('DB_HOST_MODEL'),
        port=os.getenv('DB_PORT_MODEL'),
        user=os.getenv('DB_USER_MODEL'),
        password=os.getenv('DB_PASSWORD_MODEL'),
        dbname=os.getenv('DB_NAME_MODEL')
    )

    # Tutup koneksi setelah pengujian jika ada
    if connection:
        connection.close()