import re
from nltk.corpus import wordnet

WN_NOUN = "n"
WN_VERB = "v"
WN_ADJECTIVE = "a"
WN_ADJECTIVE_SATELLITE = "s"
WN_ADVERB = "r"


def convert(word, from_pos, to_pos, with_frameset=False):
    """
        Transform a word from a given part of speech to another one
        :param word: word to be converted
        :param from_pos: part of speech to convert from
        :param to_pos: part of speech to convert to
        :param with_frameset: output the frameset id or just the word
        :return: the converted word
    """
    result = _convert(word, from_pos, to_pos)

    if len(result) > 0:
        if with_frameset:
            return re.sub("\..{1}\.", "-", result[0][0].__self__._synset._name)
        else:
            return result[0][0].__self__._synset._name.split(".")[0]
    else:
        return word


def _convert(word, from_pos, to_pos):
    """
    Get a list of probable forms for a word with the specified part of speech
    :param word: word to be converted
    :param from_pos: part of speech to convert from
    :param to_pos: part of speech to convert to
    :return: a list of most probable words with the specified part of speech as [(Lemma instance, probability)]
    """
    # word not specified
    if word is None:
        return []

    synsets = wordnet.synsets(word, pos=from_pos)

    # word not found
    if not synsets:
        return []

    # get all lemmas of the word (consider "a" and "s" equivalent)
    lemmas = [l for s in synsets
              for l in s._lemmas
              if s._name.split(".")[1] == from_pos
              or from_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)
              and s._name.split(".")[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)]

    # get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]

    # filter only the desired pos (consider "a" and "s" equivalent)
    related_noun_lemmas = [l for drf in derivationally_related_forms
                           for l in drf[1]
                           if l._synset._name.split(".")[1] == to_pos
                           or to_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)
                           and l._synset._name.split(".")[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)]

    # extract the words from the lemmas
    words = [l.name for l in related_noun_lemmas]
    len_words = len(words)

    # build the result in the form of a list containing tuples (word, probability)
    result = [(w, float(words.count(w)) / len_words) for w in set(words)]
    result.sort(key=lambda w: -w[1])

    # return all the possibilities sorted by probability
    return result


if __name__ == "__main__":
    print("death")
    res = _convert("death", WN_NOUN, WN_VERB)
    for res_i in res:
        print((res_i[0].__self__._synset._name.replace(".v.", "-"), res_i[1]))
    print((convert("death", WN_NOUN, WN_VERB)))

    print(("\n", "person"))
    res = _convert("person", WN_NOUN, WN_VERB)
    for res_i in res:
        print((res_i[0].__self__._synset._name.replace(".v.", "-"), res_i[1]))
    print((convert("person", WN_NOUN, WN_VERB)))

    print(("\n", "organization"))
    res = _convert("organization", WN_NOUN, WN_VERB)
    for res_i in res:
        print((res_i[0].__self__._synset._name.replace(".v.", "-"), res_i[1]))
    print((convert("organization", WN_NOUN, WN_VERB)))

    print(("\n", "story"))
    res = _convert("story", WN_NOUN, WN_VERB)
    for res_i in res:
        print((res_i[0].__self__._synset._name.replace(".v.", "-"), res_i[1]))
    print((convert("story", WN_NOUN, WN_VERB)))

    print(("\n", "boring"))
    res = _convert("boring", WN_ADJECTIVE, WN_NOUN)
    for res_i in res:
        print((res_i[0].__self__._synset._name.replace(".n.", "-"), res_i[1]))
    print((convert("boring", WN_ADJECTIVE, WN_NOUN)))

    print(("\n", "trouble"))
    res = _convert("trouble", WN_NOUN, WN_ADJECTIVE)
    for res_i in res:
        print((res_i[0].__self__._synset._name.replace(".a.", "-"), res_i[1]))
    print((convert("trouble", WN_NOUN, WN_ADJECTIVE)))

    print(("\n", "solve"))
    res = _convert("solve", WN_VERB, WN_ADJECTIVE_SATELLITE)
    for res_i in res:
        print((res_i[0].__self__._synset._name.replace(".s.", "-"), res_i[1]))
    print((convert("solve", WN_VERB, WN_ADJECTIVE_SATELLITE)))

    print(("\n", "think"))
    res = _convert("think", WN_VERB, WN_ADJECTIVE)
    for res_i in res:
        print((res_i[0].__self__._synset._name.replace(".a.", "-"), res_i[1]))
    print((convert("think", WN_VERB, WN_ADJECTIVE)))
