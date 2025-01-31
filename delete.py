import os
import shutil

# Define the paths
files_to_delete = [
    "trainer.yml",
    "attendance.csv",
    "registered_users.csv"
]
directory_to_delete = "dataset"

# Delete specified files
for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted successfully.")
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except PermissionError:
        print(f"Permission denied to delete the file '{file_path}'.")
    except Exception as e:
        print(f"Error occurred while deleting the file '{file_path}': {e}")

# Delete the directory and its contents
try:
    shutil.rmtree(directory_to_delete)
    print(f"Directory '{directory_to_delete}' has been deleted successfully.")
except FileNotFoundError:
    print(f"Directory '{directory_to_delete}' not found.")
except PermissionError:
    print(f"Permission denied to delete the directory '{directory_to_delete}'.")
except Exception as e:
    print(f"Error occurred while deleting the directory '{directory_to_delete}': {e}")
