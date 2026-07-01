import sys
import os
sys.path.insert(0, '.')
from pathlib import Path

directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'develop_documents')
doc_dir = Path(directory)
print('Resolved path:', doc_dir.resolve())
print('Exists:', doc_dir.exists())
print('Is dir:', doc_dir.is_dir() if doc_dir.exists() else 'N/A')

count = 0
for root, dirs, files in os.walk(doc_dir):
    for f in files:
        if f.endswith(('.docx', '.pdf', '.doc', '.txt')):
            count += 1
            if count <= 5:
                print(f'Found file: {os.path.join(root, f)}')

print(f'Total supported files: {count}')
