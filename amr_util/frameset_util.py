import numpy as np

from amr_util import word_embeddings_util
from data_extraction import frameset_parser

propbank_frames = None
nombank_frames = None


def init_propbank_frames():
    global propbank_frames

    propbank_frames = frameset_parser.load_frames("propbank")


def init_nombank_frames():
    global nombank_frames

    nombank_frames = frameset_parser.load_frames("nombank")


def get_propbank_frame(token):
    global propbank_frames

    return propbank_frames.get(token, None)


def compute_best_roleset(token, token_context, source):
    """
    Determine the roleset best matching a word, given the context of that word
    :param token: word for which to determine the best matching roleset
    :param token_context: the words that are related to the given word in the AMR graph
    :return: the beset roleset and the matching degree
    """

    global propbank_frames

    token_frameset = get_propbank_frame(token)

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
    similarities = [(r, word_embeddings_util.compute_similarity_to_sentence(token, r.description)) for r in roles]
    if len(similarities):
        return max(similarities, key=(lambda role_sim_pair: role_sim_pair[1]))
    else:
        return None, -1


if __name__ == "__main__":
    emb_dim = 200
    word_embeddings_util.init_embeddings_matrix(emb_dim)

    init_propbank_frames()
    init_nombank_frames()

    roleset, sim = compute_best_roleset("run", ["factory"], "propbank")
    print roleset.id, roleset.name, sim

    roleset, sim = compute_best_roleset("run", ["marathon"], "propbank")
    print roleset.id, roleset.name, sim
