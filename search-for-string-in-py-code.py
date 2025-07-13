import os

def search_py_files_for_string(root_dir, search_string):
    matches = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    if search_string in f.read():
                        matches.append(file_path)
    return matches

# Example usage
if __name__ == "__main__":
    matches = search_py_files_for_string('.', 'short-coco-sprite')
    for match in matches:
        print(match)
