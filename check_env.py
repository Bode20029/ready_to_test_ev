import os
from dotenv import load_dotenv

# Attempt to load the .env file
load_dotenv()

# List of expected environment variables
expected_vars = [
    'LINE_NOTIFY_TOKEN',
    'MONGO_URI',
    'DB_NAME',
    'COLLECTION_NAME'
]

# Check each variable
for var in expected_vars:
    value = os.getenv(var)
    if value:
        print(f"{var} is set to: {value}")
    else:
        print(f"{var} is not set!")

# Additional check for python-dotenv loading
print("\nChecking if .env file is being loaded:")
test_var = os.getenv('TEST_VAR', 'NOT_FOUND')
print(f"TEST_VAR = {test_var}")

if __name__ == "__main__":
    print("Run this script to check your environment variables.")
    print("Make sure to add TEST_VAR=LOADED to your .env file before running.")