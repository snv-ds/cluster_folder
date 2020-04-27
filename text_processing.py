from sklearn.feature_extraction.text import TfidfVectorizer
from re import findall

class TextProcessor:
    def __init__(self, corpus):
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer()

    def _preprocess(self, pattern="[^.,:;!?_\-()\+\|[0-9]]*"):
        for ind, row in enumerate(self.corpus):
            self.corpus[ind] = ''.join(findall(pattern, row.lower()))

    def vectorize(self):
        self._preprocess()
        return self.vectorizer.fit_transform(self.corpus)

    @staticmethod
    def collect_texts(texts):
        result = ''
        for text in texts:
            result += text.lower()
        return result

    # @staticmethod
    # def get_embedding(matrix):
    #     result = ''
    #     for text in texts:
    #         result += text.lower()
    #     return result
