import pickle
from os import path

from keras.preprocessing.text import Tokenizer

from data_extraction import dataset_loader
from definitions import TOKENIZER_PATH


def generate_tokenizer():
    train_data_sentences = [d.sentence for d in dataset_loader.read_data("training", cache=True)]
    dev_data_sentences = [d.sentence for d in dataset_loader.read_data("dev", cache=True)]
    test_data_sentences = [d.sentence for d in dataset_loader.read_data("test", cache=True)]

    sentences = train_data_sentences + dev_data_sentences + test_data_sentences

    tokenizer = Tokenizer(filters="", lower=True, split=" ")
    tokenizer.fit_on_texts(sentences)

    pickle.dump(tokenizer, open(TOKENIZER_PATH, "wb"))


def get_tokenizer():
    if not path.isfile(TOKENIZER_PATH):
        generate_tokenizer()

    return pickle.load(open(TOKENIZER_PATH, "rb"), encoding='latin1')


tokenizer = get_tokenizer()


def get_word_index_map():
    tokenizer = get_tokenizer()
    return tokenizer.word_index


def get_index_word_map():
    tokenizer = get_tokenizer()
    return tokenizer.index_word


def get_no_word_index():
    return len(get_word_index_map()) + 1


def text_to_sequence(sentence):
    return tokenizer.texts_to_sequences([sentence])[0]


def sequence_to_text(sequence):
    return tokenizer.sequences_to_texts([sequence])[0]


if __name__ == "__main__":
    word_index_map = get_word_index_map()
    print(("Words to indices map: %d" % len(word_index_map)))

    index_word_map = get_index_word_map()
    print(("Indices to words map: %d" % len(index_word_map)))

    index_word_map_copy = {v: k for k, v in list(word_index_map.items())}
    print((cmp(index_word_map, index_word_map_copy)))

    sentence = "He went to the store to buy wood for a new fence ."
    tokens_sequence = text_to_sequence(sentence)

    print(("Original sentence:", sentence))
    print(("Tokens sequence:", tokens_sequence))
    print(("Reconstructed sentence:", sequence_to_text(tokens_sequence)))
