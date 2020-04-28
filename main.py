from utils import get_directory_content, iter_dir, parse_arguments
from text_processing import TextProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sys import exit
from text_clustering import get_simularity


if __name__ == '__main__':
    args = parse_arguments()
    result = iter_dir(args.path)
    file_names = list(result.keys())
    text_processor = TextProcessor(list(result.values()))
    # print(text_processor.get_phrase_embedding(3))
    get_simularity(text_processor.vectorized, token=file_names)
