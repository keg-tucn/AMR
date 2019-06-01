import re
import numpy as np
from nltk.tokenize.stanford import StanfordTokenizer

from definitions import GLOVE_EMBEDDINGS, STANFORD_POSTAGGER_JAR
import tokenizer_util

embeddings_matrix = None
full_embeddings_matrix_dict = None
current_embedding_dim = None
word_index_map = None
tokenizer = None


def init_embeddings_matrix(embedding_dim):
    global embeddings_matrix, full_embeddings_matrix_dict, current_embedding_dim, word_index_map, tokenizer

    word_index_map = tokenizer_util.get_word_index_map()
    current_embedding_dim = embedding_dim
    embeddings_matrix, full_embeddings_matrix_dict = _load_embeddings_matrix(word_index=word_index_map,
                                                                             embedding_dim=embedding_dim)
    tokenizer = StanfordTokenizer(path_to_jar=STANFORD_POSTAGGER_JAR)


def get_embeddings_matrix(embedding_dim):
    global embeddings_matrix, current_embedding_dim

    if current_embedding_dim != embedding_dim:
        init_embeddings_matrix(embedding_dim)

    return embeddings_matrix


def get_token_embedding(token):
    global embeddings_matrix

    token_index = word_index_map.get(token)
    return embeddings_matrix[token_index, :]


def get_token_embeddings(token):
    global full_embeddings_matrix_dict

    return full_embeddings_matrix_dict.get(token)


def compute_cosine_similarity(token_1, token_2):
    emb_1 = get_token_embeddings(token_1)
    emb_2 = get_token_embeddings(token_2)

    if emb_1 is not None and emb_2 is not None:
        # Compute the dot product between emb_1 and emb_2
        dot_emb_1_emb_2 = np.dot(emb_1, emb_2)
        # Compute the L2 norm of emb_1
        norm_emb_1 = np.sqrt(np.sum(emb_1 ** 2))
        # Compute the L2 norm of emb_2
        norm_emb_2 = np.sqrt(np.sum(emb_2 ** 2))
        # Compute the cosine similarity
        cosine_similarity = dot_emb_1_emb_2 / np.dot(norm_emb_1, norm_emb_2)

        return cosine_similarity
    else:
        return 0


def compute_cosine_similarity_vectors(emb_1, emb_2):
    if emb_1 is not None and emb_2 is not None:
        # Compute the dot product between emb_1 and emb_2
        dot_emb_1_emb_2 = np.dot(emb_1, emb_2)
        # Compute the L2 norm of emb_1
        norm_emb_1 = np.sqrt(np.sum(emb_1 ** 2))
        # Compute the L2 norm of emb_2
        norm_emb_2 = np.sqrt(np.sum(emb_2 ** 2))
        # Compute the cosine similarity
        cosine_similarity = dot_emb_1_emb_2 / np.dot(norm_emb_1, norm_emb_2)

        return cosine_similarity
    else:
        return 0


def compoute_most_similar(token, candidates):
    best_similarity = -1
    most_similar = None

    for candidate in candidates:
        similarity = compute_cosine_similarity(token, candidate)
        if similarity > best_similarity:
            best_similarity = similarity
            most_similar = candidate

    return most_similar, best_similarity


def compute_similarity_to_sentence(token, sentence):
    global tokenizer

    token_to_token_similarity = []
    sentence_tokens = tokenizer.tokenize(sentence)
    for sentence_token in sentence_tokens:
        token_to_token_similarity.append(compute_cosine_similarity(token, sentence_token.encode("utf-8")))
    if len(token_to_token_similarity):
        return max(token_to_token_similarity)
    else:
        return -1


def precompute_cosine_similarities():
    global full_embeddings_matrix_dict

    cross_cosine_similarities = {}
    for (key_1, value_1), i in zip(full_embeddings_matrix_dict.items(),
                                   range(len(full_embeddings_matrix_dict.items()))):
        for (key_2, value_2), j in zip(full_embeddings_matrix_dict.items(),
                                       range(len(full_embeddings_matrix_dict.items()))):
            if i % 10000 == 0 and j % 10000 == 0:
                print "%d %d" % (i, j)
            cross_cosine_similarities[(key_1, key_2)] = compute_cosine_similarity_vectors(value_1, value_2)

    return cross_cosine_similarities


def _load_embeddings_matrix(word_index, embedding_dim):
    special_cases_re = re.compile("""^([a-z])+-(?:entity|quantity)$""")

    embeddings_index = {}

    emb_file = open(GLOVE_EMBEDDINGS + "/" + "glove.6B.{}d.txt".format(embedding_dim))
    for line in emb_file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    emb_file.close()

    print("Found %s word vectors." % len(embeddings_index))

    embeddings_matrix = np.zeros((len(word_index) + 2, embedding_dim))
    not_found = []

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embeddings_matrix[i] = embedding_vector
        else:
            match = re.match(special_cases_re, word)
            if match:
                print "Embedding match for {}".format(word)
                embedding_vector = embeddings_index.get(match.group(1))
            else:
                not_found.append(word)

    print "First 4 not found: {}".format(not_found[0:4])
    return embeddings_matrix, embeddings_index


if __name__ == "__main__":
    emb_dim = 200
    init_embeddings_matrix(emb_dim)

    '''
    father = get_token_embedding("father")
    mother = get_token_embedding("mother")
    ball = get_token_embedding("ball")
    crocodile = get_token_embedding("crocodile")
    france = get_token_embedding("france")
    italy = get_token_embedding("italy")
    paris = get_token_embedding("paris")
    rome = get_token_embedding("rome")

    print "cosine_similarity(father, mother) = %f" % compute_cosine_similarity(father, mother)
    print "cosine_similarity(ball, crocodile) = %f" % compute_cosine_similarity(ball, crocodile)
    print "cosine_similarity(france - paris, rome - italy) = %f" % compute_cosine_similarity(france - paris,
                                                                                             rome - italy)
    print "cosine_similarity(france - paris, italy - rome) = %f" % compute_cosine_similarity(france - paris,
                                                                                             italy - rome)
    '''

    print "cosine similarity (ball, mother): %f" % compute_cosine_similarity("ball", "mother")
    print "cosine similarity (woman, mother): %f" % compute_cosine_similarity("woman", "mother")
    print "cosine similarity (make, made): %f" % compute_cosine_similarity("make", "made")
    print "cosine similarity (do, did): %f" % compute_cosine_similarity("do", "did")
