from utils import get_directory_content, iter_dir, parse_arguments
from text_processing import TextProcessor
from sys import exit


if __name__ == '__main__':
    args = parse_arguments()
    result = iter_dir(args.path)
    file_names = list(result.keys())
    text_processor = TextProcessor(list(result.values()))
    text_processor._preprocess()
    texts = text_processor.corpus
    print(texts)
