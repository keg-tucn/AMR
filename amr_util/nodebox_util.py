import en


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


def simplify_noun(word):
    return en.noun.singular(word)


def simplify_verb(word):
    return en.verb.present(word)


def is_noun(word):
    return en.is_noun(word)


def is_verb(word):
    return en.is_verb(word)


def is_adjective(word):
    return en.is_adjective(word)


def is_adverb(word):
    return en.is_adverb(word)


if __name__ == "__main__":
    print simplify_word("fellow")
    print simplify_word("particles")
    print simplify_word("negotiated")
    print simplify_word("played")
    print simplify_word("-)")

    print

    print is_noun("particles")
    print is_noun(simplify_word("particles"))
    print is_verb("negotiated")
    print is_verb(simplify_word("negotiated"))
