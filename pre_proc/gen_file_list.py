import os

# Replace 'your_directory_path' with the path to the directory you want to scan
directory_path = '/home/jc/dataset/20240113_0301/Masks/'

# List all files in the specified directory
file_names = [file for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
# Sorting the list for better readability
file_names.sort()

output_file = '/home/jc/dataset/20240113_0301/test.txt'
with open(output_file, 'w') as file:
    for name in file_names:
        file.write(name + '\n')