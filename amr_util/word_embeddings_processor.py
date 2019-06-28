import numpy as np
from nltk.tokenize.stanford import StanfordTokenizer

from data_extraction import word_embeddings_reader
from definitions import STANFORD_POSTAGGER_JAR

tokenizer = StanfordTokenizer(path_to_jar=STANFORD_POSTAGGER_JAR)


def compute_cosine_similarity(token_1, token_2):
    emb_1 = word_embeddings_reader.get_token_embedding_from_full(token_1)
    emb_2 = word_embeddings_reader.get_token_embedding_from_full(token_2)

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


if __name__ == "__main__":
    emb_dim = 200
    word_embeddings_reader.init_embeddings_matrix(emb_dim)

    father = word_embeddings_reader.get_token_embedding_from_reduced("father")
    mother = word_embeddings_reader.get_token_embedding_from_reduced("mother")
    ball = word_embeddings_reader.get_token_embedding_from_reduced("ball")
    crocodile = word_embeddings_reader.get_token_embedding_from_reduced("crocodile")
    france = word_embeddings_reader.get_token_embedding_from_reduced("france")
    italy = word_embeddings_reader.get_token_embedding_from_reduced("italy")
    paris = word_embeddings_reader.get_token_embedding_from_reduced("paris")
    rome = word_embeddings_reader.get_token_embedding_from_reduced("rome")

    print "cosine_similarity(father, mother) = %f" % compute_cosine_similarity_vectors(father, mother)
    print "cosine_similarity(ball, crocodile) = %f" % compute_cosine_similarity_vectors(ball, crocodile)
    print "cosine_similarity(france - paris, rome - italy) = %f" % compute_cosine_similarity_vectors(france - paris,
                                                                                                     rome - italy)
    print "cosine_similarity(france - paris, italy - rome) = %f" % compute_cosine_similarity_vectors(france - paris,
                                                                                                     italy - rome)

    print "cosine similarity (ball, mother): %f" % compute_cosine_similarity("ball", "mother")
    print "cosine similarity (woman, mother): %f" % compute_cosine_similarity("woman", "mother")
    print "cosine similarity (make, made): %f" % compute_cosine_similarity("make", "made")
    print "cosine similarity (do, did): %f" % compute_cosine_similarity("do", "did")
    print "cosine similarity (do, did): %f" % compute_cosine_similarity("do-01", "did")
