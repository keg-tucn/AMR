import functools
import re

import numpy as np

from amr_util import tokenizer_util
from definitions import GLOVE_EMBEDDINGS

# embedding vectors only for the tokens recognized by the tokenizer in the data set
reduced_embeddings_matrix = None
# complete dictionary of word embeddings
full_embeddings_matrix_dict = None
current_embedding_dim = None
word_index_map = None
tokenizer = None


def init_embeddings_matrix(embedding_dim):
    global reduced_embeddings_matrix, full_embeddings_matrix_dict, current_embedding_dim, word_index_map, tokenizer

    word_index_map = tokenizer_util.get_word_index_map()
    current_embedding_dim = embedding_dim
    reduced_embeddings_matrix, full_embeddings_matrix_dict = _load_embeddings_matrix(word_index=word_index_map,
                                                                                     embedding_dim=embedding_dim)


def get_embeddings_matrix(embedding_dim):
    global reduced_embeddings_matrix, current_embedding_dim

    if current_embedding_dim != embedding_dim:
        init_embeddings_matrix(embedding_dim)

    return reduced_embeddings_matrix


def get_token_embedding_from_reduced(token):
    global reduced_embeddings_matrix

    token_index = word_index_map.get(token)
    return reduced_embeddings_matrix[token_index, :]


def get_token_embedding_from_full(token):
    global full_embeddings_matrix_dict

    return full_embeddings_matrix_dict.get(token)

@functools.lru_cache(maxsize=5)
def read_glove_embeddings_from_file(embedding_dim):
    """
    Read glove embedding from file corresponding to embedding_dim (can be 50, 100, 200, 300)
    Return dictionary of word -> glove_embedding
    """
    embeddings_index = {}

    emb_file = open(GLOVE_EMBEDDINGS + "/" + "glove.6B.{}d.txt".format(embedding_dim), encoding="utf-8")
    for line in emb_file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    emb_file.close()
    return embeddings_index


def _load_embeddings_matrix(word_index, embedding_dim):
    special_cases_re = re.compile("""^([a-z])+-(?:entity|quantity)$""")

    embeddings_index = read_glove_embeddings_from_file(embedding_dim)

    print(("Found %s word vectors." % len(embeddings_index)))

    embeddings_matrix = np.zeros((len(word_index) + 2, embedding_dim))
    not_found = []

    for word, i in list(word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embeddings_matrix[i] = embedding_vector
        else:
            match = re.match(special_cases_re, word)
            if match:
                print(("Embedding match for {}".format(word)))
                embedding_vector = embeddings_index.get(match.group(1))
            else:
                not_found.append(word)

    print(("First 4 not found: {}".format(not_found[0:4])))
    return embeddings_matrix, embeddings_index
