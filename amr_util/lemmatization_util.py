from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize(word):
    pos = get_wordnet_pos(word)

    if pos is not None:
        return wordnet_lemmatizer.lemmatize(word, pos)
    else:
        return wordnet_lemmatizer.lemmatize(word)


if __name__ == "__main__":
    print lemmatize("particles")
    print lemmatize("negotiated")
    print lemmatize("was")
    print lemmatize("played")
    print lemmatize("plays")
    print lemmatize("-)")
    print lemmatize("bats")
    print lemmatize("are")
    print lemmatize("feet")
