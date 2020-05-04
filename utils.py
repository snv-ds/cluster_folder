from textract import process as process_doc
from docx2txt import process as process_docx
from os import path, listdir
from argparse import ArgumentParser
from text_processor.text_processing import TextProcessor
from text_processor.text_clustering import visualize_simularity, cluster_docs, print_clusters


def get_file_content(file_path):
    """
    Function for getting content of text file.
    Parameters:
        file_path (str): path of text files
    Returns:
        (str): content of file
    """
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
    """
    Function for recursive walkthrough over directory with text files.
    Parameters:
        dir_path (str): path of directory with text files
        result (dict): dictionary with keys = file names, and values texts of files, which is updating
    """
    children = [path.join(dir_path, child) for child in listdir(dir_path)]
    files = filter(path.isfile, children)
    result[dir_path] = list(files)

    directories = filter(path.isdir, children)
    for directory in directories:
        get_directory_content(directory, result)


def iter_dir(file_path):
    """
    Function for iterating directory and getting content of text files.
    Parameters:
        file_path (str): path of directory with text files
    Returns:
        result (dict): dictionary with keys = file names, and values texts of files
    """
    files_per_path = {}
    result = {}
    get_directory_content(file_path, files_per_path)
    for files in files_per_path.values():
        for file in files:
            content = get_file_content(file)
            if content:
                result[file] = content
    return result


def parse_arguments():
    """
    Function for parsing arguments of CLI.
    Returns:
        (argparse.ArgumentParser): parsed arguments
    """
    parser = ArgumentParser(__doc__)
    parser.add_argument("--path", "-p", help="Folder for text walk through.", default="./")
    parser.add_argument("--verbose", "-v", help="Display clusterisation of files", default=False)
    return parser.parse_args()


def main():
    """
    Main function of program.
    """
    args = parse_arguments()
    result = iter_dir(args.path)
    file_names = list(result.keys())
    text_processor = TextProcessor(list(result.values()))
    if args.verbose:
        visualize_simularity(text_processor.vectorized, token=file_names)
    clusters = cluster_docs(text_processor.vectorized, file_names)
    print_clusters(clusters, file_names)
