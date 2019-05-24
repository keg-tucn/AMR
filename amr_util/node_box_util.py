import en


def simplify_word(word):
    try:
        if word != simplify_verb(word):
            return simplify_verb(word)

        if word != simplify_noun(word):
            return simplify_noun(word)
    except:
        return word


def simplify_noun(word):
    return en.noun.singular(word)


def simplify_verb(word):
    return en.verb.present(word)


if __name__ == "__main__":
    print en.is_noun("particles")
    print en.is_verb("negotiated")

    print simplify_word("particles")
    print simplify_word("negotiated")
    print simplify_word("-)")
