import io
from pymagnitude import *

from definitions import PROJECT_ROOT_DIR

MAGNITUDE_PATH = PROJECT_ROOT_DIR + '/resources/magnitude'
MAGNITUDE_PATH_LIGHT = MAGNITUDE_PATH + '/light'
MAGNITUDE_PATH_MEDIUM = MAGNITUDE_PATH + '/medium'
GLOVE_PATH = MAGNITUDE_PATH + '/{}/glove.6B.{}d.magnitude'
FASTTEXT_WIKI_SUBWORD_PATH = MAGNITUDE_PATH + '/{}/wiki-news-300d-1M-subword.magnitude'

def get_magnitude_glove_vectors(dim: int, weight='medium'):
    """
    Return magnitude glove vectors
    Parameter:
        dim: vector dimensions
        weight: light, medium, heavy (by default medium)
    """
    glove_file_path = GLOVE_PATH.format(weight, dim)
    vectors = Magnitude(glove_file_path)
    return vectors


def get_magnitude_fasttext_vectors(weight='medium'):
    """
    Return magnitude fasttext vectors
    Parameter:
        weight: light, medium, heavy (by default medium)
    """
    fasttext_file_path = FASTTEXT_WIKI_SUBWORD_PATH.format(weight)
    vectors = Magnitude(fasttext_file_path)
    return vectors

