class Vocab:
    def __init__(self, w2i):
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in list(w2i.items())}

    @classmethod
    def from_list(cls, words):
        w2i = {}
        idx = 0
        for word in words:
            w2i[word] = idx
            idx += 1
        return Vocab(w2i)

    @classmethod
    def from_file(cls, vocab_fname):
        words = []
        with open(vocab_fname) as fh:
            for line in fh:
                line.strip()
                word, count = line.split()
                words.append(word)
        return Vocab.from_list(words)

    def size(self):
        return len(list(self.w2i.keys()))

    def get_index_or_add(self, word):
        if word not in list(self.w2i.keys()):
            idx = self.size()
            self.w2i[word] = idx
            self.i2w[idx] = word
            return idx
        else:
            return self.w2i[word]


def word_sentence_to_vocab_index(sentence_words, word_vocab):
    return [word_vocab.get_index_or_add(x) for x in sentence_words]
