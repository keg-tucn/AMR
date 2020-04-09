from pattern.en import tag
from pattern.en import wordnet
from pattern.en import singularize
from pattern.en import conjugate, PRESENT, SG

# tags
NOUN = 0
VERB = 1
ADJ = 2
ADV = 3


def simplify_word(word):
    try:
        if word != simplify_verb(word):
            return simplify_verb(word)
        else:
            return word
    except:
        try:
            if word != simplify_noun(word):
                return simplify_noun(word)
        except:
            return word
        return word


# pattern.en tag has output of the form [('has', 'VBZ')]
# could be used on more words at the same time (better accuracy, maybe faster)
def get_tag(word):
    tagger_output = tag(word)
    return tagger_output[0][1]


# inspired from here: https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
def get_wordnet_pos(word):
    treebank_tag = get_tag(word)
    if treebank_tag.startswith('J'):
        return ADJ
    elif treebank_tag.startswith('V'):
        return VERB
    elif treebank_tag.startswith('N'):
        return NOUN
    elif treebank_tag.startswith('R'):
        return ADV
    else:
        return ''


def simplify_noun(word):
    return singularize(word)


# taken from https://stackoverflow.com/questions/3753021/using-nltk-and-wordnet-how-do-i-convert-simple-tense-verb-into-its-present-pas
def simplify_verb(word):
    # why put it in present simple and not use lemma?
    return conjugate(verb=word, tense=PRESENT, number=SG)


def is_noun(word):
    return get_wordnet_pos(word) == NOUN


def is_verb(word):
    return get_wordnet_pos(word) == VERB


def is_adjective(word):
    return get_wordnet_pos(word) == ADJ


def is_adverb(word):
    return get_wordnet_pos(word) == ADV


if __name__ == "__main__":
    print((simplify_word("fellow")))
    print((simplify_word("particles")))
    print((simplify_word("negotiated")))
    print((simplify_word("played")))
    print((simplify_word("-)")))

    print()

    print((is_noun("particles")))
    print((is_noun(simplify_word("particles"))))
    print((is_verb("negotiated")))
    print((is_verb(simplify_word("negotiated"))))
