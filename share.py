import os

def create_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder if it doesn't exist
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def delete_file(file_path):
    # Check if the file exists
    if os.path.exists(file_path):
        # Delete the file
        os.remove(file_path)
        print(f'The file {file_path} has been deleted.')
    else:
        print(f'The file {file_path} does not exist.')
