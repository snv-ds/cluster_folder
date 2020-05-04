from sklearn.feature_extraction.text import TfidfVectorizer
from re import findall


class TextProcessor:
    """
    Class for preprocessing over texts.
    Parameters:
        corpus (list): List of all texts, each value of list is text of file.
    """
    def __init__(self, corpus):
        self.corpus = corpus
        self._vectorized = self._vectorize()

    @property
    def vectorized(self):
        return self._vectorized

    def _preprocess(self, pattern="[^.,:;!?_\-()\+\|[0-9]]*"):  # pattern for all words except numbers and punctuation
        """
        The function filter text of given pattern.
        Parameters:
            pattern (str): pattern for string of regular expression
        Returns:
            self.corpus(list): all matched texts
        """
        for ind, row in enumerate(self.corpus):
            self.corpus[ind] = ''.join(findall(pattern, row.lower()))

    def _vectorize(self):
        """
         Get vectorized implementation of texts.
         Returns:
             _vectorized (scipy.sparse.csr_matrix): sparse matrix of tf-idf presentation of all texts
         """
        self._preprocess()
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(self.corpus)

    def get_phrase_embedding(self, index):
        """
        Get particular embedding text(function needed for testing).
        Parameters:
            index (int): index of file
        Returns:
            _vectorized (scipy.sparse.csr_matrix): row of sparse matrix of tf-idf presentation for particular text
        """
        if 0 <= index < len(self.corpus):
            return self.vectorized[index, :]

    @staticmethod
    def lower_texts(texts):
        """
        Get particular embedding text.
        Parameters:
            texts (list): list of texts
        Returns:
            result (list): list of all texts in lowercase
        """
        result = ''
        for text in texts:
            result += text.lower()
        return result
