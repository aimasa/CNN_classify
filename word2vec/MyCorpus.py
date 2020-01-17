from gensim import utils


class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __init__(self, list_text):
        self.list_text = list_text

    def __iter__(self):
        for line in self.list_text:
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(str(line))
