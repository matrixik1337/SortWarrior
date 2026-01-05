import os
import re
from pathlib import Path

path = input("Enter path to annotations directory: ")
directory = Path(path)

old_word = input("Enter old id: ")
new_word = input("Enter new id: ")

for file_path in directory.rglob('*'):
    if file_path.is_file():
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            pattern = r'\b' + re.escape(old_word) + r'\b'
            
            new_content, count = re.subn(pattern, new_word, content)
            
            if count > 0:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(new_content)
                print(f"{file_path}: {count} changes")
                
        except (UnicodeDecodeError, PermissionError):
            continue
