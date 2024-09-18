import subprocess
import requests
import json
import os

# List of Python files to run
files = [
    '/app/py/01_nlp_analysis.py',
    '/app/py/02_pickle_to_blob.py',
    '/app/py/03_logging_to_db.py',
    '/app/serving/collect_blob.py',
    '/app/serving/prediction.py'
]

def send_slack_message(message):
    webhook_url = 'https://hooks.slack.com/services/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'
    
    payload = {
        'text': message
    }
    headers = {
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(webhook_url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
    except requests.exceptions.RequestException as e:
        print(f"Failed to send message to Slack: {e}")

# Run each file sequentially
for file in files:
    print(f"Running {file}")
    result = subprocess.run(['python', '-u', file], capture_output=True, text=True)
    
    # Print stdout and stderr
    print(f"STDOUT from {file}:")
    print(result.stdout)
    print(f"STDERR from {file}:")
    print(result.stderr)
    
    # If there is an error, send a Slack notification and stop further execution
    if result.returncode != 0:
        error_message = (
            f"Error Project NLP Risk Register \n"
            f"Error running {os.path.basename(file)}:\n"
            f"Return Code: {result.returncode}\n"
            f"STDERR: {result.stderr}"
        )
        print(error_message)
        send_slack_message(error_message)
        break  # Stop executing further scripts if there's an error

    # Run pytest after a specific script
    if file == 'C:/Project/Data Science/NLP Project/py/01_nlp_analysis.py':
        print("Running pytest after 01_nlp_analysis.py")
        pytest_result = subprocess.run(['pytest', '-v', 'C:/Project/Data Science/NLP Project/py/test_pre_predict.py'], capture_output=True, text=True)
        
        # Print pytest stdout and stderr
        print("STDOUT from pytest:")
        print(pytest_result.stdout)
        print("STDERR from pytest:")
        print(pytest_result.stderr)
        
        # If pytest fails, send a Slack notification and stop further execution
        if pytest_result.returncode != 0:
            pytest_error_message = (
                f"Pytest Error Project NLP Risk Register:\n"
                f"Return Code: {pytest_result.returncode}\n"
                f"STDERR: {pytest_result.stderr}"
            )
            print(pytest_error_message)
            send_slack_message(pytest_error_message)
            break  # Optionally stop if pytest fails
