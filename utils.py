from textract import process as process_doc
from docx2txt import process as process_docx
from os import path, listdir


def get_file_content(file_path):
    file_format = file_path.rsplit('.')[-1]
    if file_format == 'txt':
        with open(file_path, r) as txt_file:
            return txt_file.readlines()
    elif file_format == 'doc':
        return process_doc(file_path).decode('utf-8')
    elif file_format == 'docx':
        return process_docx(file_path)
    else:
        pass


def get_directory_content(dir_path, result):
    children = [path.join(dir_path, child) for child in listdir(dir_path)]
    files = filter(path.isfile, children)
    result[dir_path] = list(files)

    directories = filter(path.isdir, children)
    for directory in directories:
        get_directory_content(directory, result)



def iter_dir(file_path):
    files_per_path = {}
    result = {}
    get_directory_content(file_path, files_per_path)
    for files in files_per_path.values():
        for file in files:
            content = get_file_content(file)
            if content:
                result[file] = content
    return result
