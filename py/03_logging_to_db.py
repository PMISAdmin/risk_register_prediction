import os
import json
import psycopg2
from dotenv import load_dotenv

def load_json_from_file(file_path):
    """Load JSON data from a file that may contain multiple JSON objects."""
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return []

    with open(file_path, 'r') as file:
        content = file.read()
    
    json_objects = content.strip().split('\n')
    
    data = []
    for obj in json_objects:
        try:
            data.append(json.loads(obj))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {file_path}: {e}")
    
    return data

def delete_file_content(file_path):
    """Delete the content of a file."""
    open(file_path, 'w').close()

# Define paths to the JSON files (adjust paths if running in Docker or other environments)
base_path = os.path.join(os.getcwd(), 'app/logs/')
training_log_path = os.path.join(base_path, 'training_log.json')
re_train_log_path = os.path.join(base_path, 're_train_log.json')
valid_log_path = os.path.join(base_path, 'valid_log.json')
test_log_path = os.path.join(base_path, 'test_log.json')

# Load JSON data from each file
training_log = load_json_from_file(training_log_path)
re_train_log = load_json_from_file(re_train_log_path)
valid_log = load_json_from_file(valid_log_path)
test_log = load_json_from_file(test_log_path)

load_dotenv()

# Database connection details (consider using environment variables for sensitive data)
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_name = os.getenv('DB_NAME')

def insert_db_training(data, schema_name, table_name, file_path):
    """Insert data into the specified table in the PostgreSQL database and delete the file content."""
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        )
        cur = conn.cursor()
        
        # Construct SQL INSERT statement with schema and table
        insert_query = f"""
        INSERT INTO "{schema_name}"."{table_name}" (model_uid, model_name, training_time, training_date, macro_averaged_f1_score, roc_auc, balanced_accuracy, data_configurations)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """
        
        # Insert each item in data with field name mapping
        for item in data:
            try:
                cur.execute(insert_query, (
                    item.get('model_uid'),
                    item.get('model_name'),
                    item.get('training_time'),
                    item.get('training_date'),
                    item.get('Macro-Averaged F1 Score', None), 
                    item.get('ROC AUC', None),                    
                    item.get('Balanced Accuracy', None),         
                    item.get('data_configurations', None)        
                ))
            except Exception as e:
                print(f"Error inserting item {item}: {e}")

        # Commit changes
        conn.commit()
        
        # Close cursor and connection
        cur.close()
        conn.close()
        
        # Clear the file content after successful insertion
        delete_file_content(file_path)
        
        print(f"Data inserted successfully into {schema_name}.{table_name}. File content cleared.")
    
    except Exception as e:
        print(f"Error connecting to the database or inserting data: {e}")

def insert_db_test(data, schema_name, table_name, file_path):
    """Insert data into the specified table in the PostgreSQL database and delete the file content."""
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        )
        cur = conn.cursor()
        
        # Construct SQL INSERT statement with schema and table
        insert_query = f"""
        INSERT INTO "{schema_name}"."{table_name}" 
        (model_uid, model_name, training_time, training_date, macro_averaged_f1_score, roc_auc, 
        balanced_accuracy, classification_report, confusion_matrix, data_configurations)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        
        # Ensure 'data' is a list of dictionaries
        if not isinstance(data, list):
            raise ValueError("The data should be a list of dictionaries.")
        
        # Insert each item in data with field name mapping
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Each item in data should be a dictionary.")
            
            try:
                cur.execute(insert_query, (
                    item.get('model_uid'),
                    item.get('model_name'),
                    item.get('training_time'),
                    item.get('training_date'),
                    item.get('Macro-Averaged F1 Score'),
                    item.get('ROC AUC'),
                    item.get('Balanced Accuracy'),
                    json.dumps(item.get('classification_report', {})) if item.get('classification_report') is not None else None,
                    json.dumps(item.get('confusion_matrix', {})) if item.get('confusion_matrix') is not None else None,
                    item.get('data_configurations')
                ))
            except Exception as e:
                print(f"Error inserting item {item}: {e}")

        # Commit changes
        conn.commit()
        
        # Close cursor and connection
        cur.close()
        conn.close()
        
        # Clear the file content after successful insertion
        delete_file_content(file_path)
        
        print(f"Data inserted successfully into {schema_name}.{table_name}. File content cleared.")
    
    except Exception as e:
        print(f"Error connecting to the database or inserting data: {e}")

# Insert data into the respective tables
insert_db_training(training_log, 'classification', 'risk_register_training_log', training_log_path)
insert_db_training(re_train_log, 'classification', 'risk_register_re_train_log', re_train_log_path)
insert_db_training(valid_log, 'classification', 'risk_register_valid_log', valid_log_path)
insert_db_test(test_log, 'classification', 'risk_register_test_log', test_log_path)
