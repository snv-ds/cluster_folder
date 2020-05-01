from utils import get_directory_content, iter_dir, parse_arguments
from text_processing import TextProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sys import exit
from text_clustering import visualize_simularity, cluster_docs, print_clusters


if __name__ == '__main__':
    args = parse_arguments()
    result = iter_dir(args.path)
    file_names = list(result.keys())
    text_processor = TextProcessor(list(result.values()))
    if args.verbose:
        visualize_simularity(text_processor.vectorized, token=file_names)
    clusters = cluster_docs(text_processor.vectorized, file_names)
    print_clusters(clusters, file_names)
