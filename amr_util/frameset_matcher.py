import numpy as np

from amr_util import word_embeddings_processor
from data_extraction import frameset_parser, word_embeddings_reader


def compute_best_roleset(token, token_context, source):
    """
    Determine the roleset best matching a word, given the context of that word
    :param token: word for which to determine the best matching roleset
    :param token_context: the words that are related to the given word in the AMR graph
    :return: the beset roleset and the matching degree
    """

    token_frameset = frameset_parser.get_frameset(token, source)

    if token_frameset is not None and len(token_frameset.rolesets):
        similarities = [(r, compute_context_roleset_similarity(token_context, r)) for r in
                        token_frameset.rolesets]

        return max(similarities, key=(lambda roleset_sim_pair: roleset_sim_pair[1]))
    else:
        return None, -1


def compute_context_roleset_similarity(token_context, roleset):
    """
    Compute how similar is a given context to a given role set
    :param token_context: the list of word that represent the context of a word in a sentence
    :param roleset: a role set of the frameset of a word
    :return: the degree of similarity of the context to the roles of the roleset
    """
    token_best_role_similarity = [r[1] for r in [compute_best_role(token, roleset.roles) for token in token_context]]

    if len(token_best_role_similarity):
        return np.mean(token_best_role_similarity)
    else:
        return -1


def compute_best_role(token, roles):
    """
    Compute the role that best matches the given token
    :param token: word for which to find the best role
    :param roles: list of roles from a role set
    :return: best matching role and the similarity degree
    """
    similarities = [(r, word_embeddings_processor.compute_similarity_to_sentence(token, r.description)) for r in roles]
    if len(similarities):
        return max(similarities, key=(lambda role_sim_pair: role_sim_pair[1]))
    else:
        return None, -1


if __name__ == "__main__":
    emb_dim = 200
    word_embeddings_reader.init_embeddings_matrix(emb_dim)

    frameset_parser.init_frames()

    roleset, sim = compute_best_roleset("run", ["factory"], "propbank")
    print roleset.id, roleset.name, sim

    roleset, sim = compute_best_roleset("run", ["marathon"], "propbank")
    print roleset.id, roleset.name, sim
