import re


class Vocab:
    def __init__(self, w2i):
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in w2i.iteritems()}

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
        with file(vocab_fname) as fh:
            for line in fh:
                line.strip()
                word, count = line.split()
                words.append(word)
        return Vocab.from_list(words)

    def size(self):
        return len(self.w2i.keys())

    def get_index_or_add(self, word):
        if not word in self.w2i.keys():
            idx = self.size()
            self.w2i[word] = idx
            self.i2w[idx] = word
            return idx
        else:
            return self.w2i[word]


def read_oracle(fname, word_vocab, action_vocab):
    with file(fname) as fh:
        for line in fh:
            line = line.strip()
            ssent, sacts = re.split(r' \|\|\| ', line)
            sent = word_sentence_to_vocab_index(ssent.split(), word_vocab)
            acts = oracle_actions_to_action_index(sacts, action_vocab)
            yield (sent, acts, ssent, sacts, "")


def word_sentence_to_vocab_index(sentence_words, word_vocab):
    return [word_vocab.get_index_or_add(x) for x in sentence_words]


def oracle_actions_to_action_index(oracle_action_sequence, action_vocab):
    if '\'' in oracle_action_sequence:
        # actions format: ['SH_label', 'RL_label', 'RR_label', 'DN']
        actions = oracle_action_sequence[2:-2].split('\', \'')
    elif " " in oracle_action_sequence:
        actions = oracle_action_sequence.split()
    else:
        actions = oracle_action_sequence
    parser_actions = [AMRAction.from_oracle(x, action_vocab) for x in actions]
    return parser_actions


class AMRAction:
    def __init__(self, action, label, index):
        self.action = action
        self.label = label
        self.index = index

    def __repr__(self):
        return "action: %s label: %s index: %s" % (self.action, self.label, self.index)

    @classmethod
    def from_oracle(cls, labeled_action, va):
        split_action = labeled_action.split("_")
        action = split_action[0]
        label = None
        if len(split_action) == 2:
            label = split_action[1]
        return AMRAction(action, label, va.w2i[action])
