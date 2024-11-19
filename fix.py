import re

# Define the file path
file_path = 'requirements.txt'

# Function to modify each line
def modify_line(line):
    # Regular expression to match the pattern and capture package name and version
    match = re.match(r'([a-zA-Z0-9_-]+)=([0-9a-zA-Z.-]+)=.*', line)
    if match:
        # Rewriting to 'package==version'
        return f"{match.group(1)}=={match.group(2)}\n"
    return line  # If no match, return the line unchanged

# Open the file and process it
with open(file_path, 'r') as file:
    lines = file.readlines()

# Modify lines and write them back to the file
with open(file_path, 'w') as file:
    for line in lines:
        file.write(modify_line(line))
