import os

def check_duplicate_rows(path, filenames):
    """
    Checks for duplicate rows across multiple text files.
    
    Parameters:
    - path (str): Directory containing the text files.
    - filenames (list): List of filenames to check.
    
    Raises:
    - ValueError: If duplicate rows are found across the files.
    """
    row_set = set()  # Set to store unique rows and check duplicates
    
    for filename in filenames:
        file_path = os.path.join(path, filename)
        
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r') as f:
            for line_number, line in enumerate(f, start=1):
                stripped_line = line.strip()  # Remove surrounding whitespace
                if stripped_line in row_set:
                    raise ValueError(f"Duplicate row found: '{stripped_line}' in file {filename}, line {line_number}")
                row_set.add(stripped_line)
                
    print("No duplicate rows found across files.")
