from sklearn.feature_extraction.text import TfidfVectorizer
from re import findall

class TextProcessor:
    def __init__(self, corpus):
        self.corpus = corpus
        self._vectorized = self._vectorize()

    @property
    def vectorized(self):
        return self._vectorized

    def _preprocess(self, pattern="[^.,:;!?_\-()\+\|[0-9]]*"):
        for ind, row in enumerate(self.corpus):
            self.corpus[ind] = ''.join(findall(pattern, row.lower()))

    def _vectorize(self):
        self._preprocess()
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(self.corpus)

    def get_phrase_embedding(self, index):
        if 0 <= index < len(self.corpus):
            return self.vectorized[index, :]

    @staticmethod
    def collect_texts(texts):
        result = ''
        for text in texts:
            result += text.lower()
        return result
