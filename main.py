import subprocess
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# List of Python files to run
files = [
    '/app/py/01_nlp_analysis.py',
    '/app/py/test_pre_predict.py',
    '/app/py/02_pickle_to_blob.py',
    '/app/py/03_logging_to_db.py',
    '/app/serving/collect_blob.py',
    '/app/serving/prediction.py'
]

def send_slack_message(message):
    """Send a message to Slack."""
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')

    if not webhook_url:
        print("Webhook URL not found in .env file.")
        return

    payload = {'text': message}
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(webhook_url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
    except requests.exceptions.RequestException as e:
        print(f"Failed to send message to Slack: {e}")

def run_python_file(file):
    """Run a Python file and return the result."""
    print(f"Running {file}")
    result = subprocess.run(['python', '-u', file], capture_output=True, text=True)

    # Print stdout and stderr
    print(f"STDOUT from {file}:")
    print(result.stdout)
    print(f"STDERR from {file}:")
    print(result.stderr)

    return result

# Run each file sequentially
for file in files:
    result = run_python_file(file)
    
    # If there is an error, send a Slack notification and stop further execution
    if result.returncode != 0:
        error_message = (
            f"Error in Project NLP Risk Register\n"
            f"Error running {os.path.basename(file)}:\n"
            f"Return Code: {result.returncode}\n"
            f"STDERR: {result.stderr.strip()}"
        )
        print(error_message)
        send_slack_message(error_message)
        break  # Stop executing further scripts if there's an error
