import os

def rename_format_in_folder(folder_path, extension):
    # List all files in the given folder
    for filename in os.listdir(folder_path):
        # Construct the full file path
        old_file = os.path.join(folder_path, filename)

        # Skip directories, only rename files
        if os.path.isfile(old_file):
            # Split the filename into name and extension
            name, _extension = os.path.splitext(filename)
            # Create a new filename with the appendix
            new_filename = f"{name}{extension}"
            new_file = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed '{filename}' to '{new_filename}'")

def rename_files_in_folder(folder_path, appendix):
    # List all files in the given folder
    for filename in os.listdir(folder_path):
        # Construct the full file path
        old_file = os.path.join(folder_path, filename)

        # Skip directories, only rename files
        if os.path.isfile(old_file):
            # Split the filename into name and extension
            name, extension = os.path.splitext(filename)
            # Create a new filename with the appendix
            new_filename = f"{name}{appendix}{extension}"
            new_file = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed '{filename}' to '{new_filename}'")

# Example usage
folder_path = '/home/jc/dataset/20240124_0204/ImagesPNG/'  # Replace with your folder path
# appendix = '_crop'  # Replace with your desired appendix
# extension = '.png'
# rename_files_in_folder(folder_path, appendix )
rename_format_in_folder(folder_path,'.png')