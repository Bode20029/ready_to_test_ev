import pymongo
from pymongo.errors import ConnectionFailure, ConfigurationError
import sys

def test_mongodb_connection(uri):
    try:
        # Attempt to connect to MongoDB
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        
        print("MongoDB connection successful!")
        
        # List available databases
        print("Available databases:")
        for db_name in client.list_database_names():
            print(f"- {db_name}")
        
        return True
    except ConfigurationError as ce:
        print(f"MongoDB Configuration Error: {ce}")
    except ConnectionFailure as cf:
        print(f"MongoDB Connection Failure: {cf}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mongodb_connection_test.py <MongoDB_URI>")
        sys.exit(1)
    
    mongodb_uri = sys.argv[1]
    test_mongodb_connection(mongodb_uri)