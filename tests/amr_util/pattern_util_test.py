from amr_util import pattern_util


def test_is_noun():
    assert pattern_util.is_noun('ana')


def test_is_verb():
    assert pattern_util.is_verb('has')


def test_is_adverb():
    assert pattern_util.is_adverb('very')


def test_is_adjective():
    assert pattern_util.is_adjective('beautiful')


def test_is_noun_plural():
    assert pattern_util.is_noun('apples')


def test_simplify_noun():
    assert pattern_util.simplify_noun('cats') == 'cat'


def test_simplify_verb():
    assert pattern_util.simplify_verb('gave') == 'gives'


if __name__ == "__main__":
    test_is_noun()
    test_is_verb()
    test_is_adverb()
    test_is_adjective()
    test_is_noun_plural()
    test_simplify_noun()
    test_simplify_verb()
    print("Everything passed")
