import os

def list_files(directory):
    """List all file names in the given directory."""
    return [f.lower() for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def compare_folders(folder1, folder2):
    """Compare files in two folders."""
    folder1_files = set(list_files(folder1))
    folder2_files = set(list_files(folder2))

    common_files = folder1_files.intersection(folder2_files)
    only_in_folder1 = folder1_files - folder2_files
    only_in_folder2 = folder2_files - folder1_files

    return common_files, only_in_folder1, only_in_folder2



if __name__ == "__main__":
    
    # Example usage:
    folder1 = '/home/jc/dataset/20240124_0204/ImagesPNG/'
    folder2 = '/home/jc/dataset/20240124_0204/Masks/'

    common, only_in_first, only_in_second = compare_folders(folder1, folder2)

    print("Files common in both folders:", common)
    if len(only_in_first) != 0:
        print("Files only in first folder:", only_in_first)
    if len(only_in_second) != 0:
        print("Files only in second folder:", only_in_second)
    
    if len(only_in_first)==0 and len(only_in_second)==0:
        print("All files are in both folders")