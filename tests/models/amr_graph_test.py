from models.amr_graph import AMR


def assert_amr_graph_dictionaries(expected_amr: AMR, parsed_amr: AMR):
    assert expected_amr.node_to_concepts == parsed_amr.node_to_concepts, \
        'Node to concepts' + str(parsed_amr.node_to_concepts) + 'should be' + str(expected_amr.node_to_concepts)
    assert expected_amr.node_to_tokens == parsed_amr.node_to_tokens, \
        'Node to tokens' + str(parsed_amr.node_to_tokens) + 'should be' + str(expected_amr.node_to_tokens)
    assert expected_amr.relation_to_tokens == parsed_amr.relation_to_tokens, \
        'Relation to tokens' + str(parsed_amr.relation_to_tokens) + 'should be' + str(expected_amr.relation_to_tokens)
    # default dict of object
    assert expected_amr == parsed_amr, \
        'Default dict' + str(parsed_amr) + 'should be' + str(expected_amr)


# amr_str = """(r / recommend-01~e.1
#                 :ARG1 (a / advocate-01~e.4
#                     :ARG1 (i / it~e.0)
#                     :manner~e.2 (v / vigorous~e.3)))"""
# sentence = """It should be vigorously advocated ."""
def test_parse_example_1():
    amr_str = """(r / recommend-01~e.1
                    :ARG1 (a / advocate-01~e.4
                        :ARG1 (i / it~e.0)
                        :manner~e.2 (v / vigorous~e.3)))"""
    parsed_amr: AMR = AMR.parse_string(amr_str)
    expected_amr: AMR = AMR()
    expected_amr.node_to_concepts = {'i': 'it', 'v': 'vigorous', 'a': 'advocate-01', 'r': 'recommend-01'}
    expected_amr.node_to_tokens = {'i': ['0'], 'v': ['3'], 'a': ['4'], 'r': ['1']}
    expected_amr.relation_to_tokens = {'manner': [('2', 'a')]}
    expected_amr['i'] = {}
    expected_amr['v'] = {}
    expected_amr['a'] = {'ARG1': [('i',)], 'manner': [('v',)]}
    expected_amr['r'] = {'ARG1': [('a',)]}
    assert_amr_graph_dictionaries(expected_amr, parsed_amr)


# ::id bolt-eng-DF-170-181103-8889109_0043.4 ::amr-annotator UCO-AMR-05 ::preferred
# amr_str = """(y2 / year~e.4
#                   :time-of~e.5 (r / recover-01~e.7
#                         :ARG1-of (e / expect-01 :polarity -~e.6))
#                   :ARG1-of (p / possible-01~e.1)
#                   :domain~e.2 (d / date-entity :year~e.4 2012~e.0))"""
def test_parse_example_with_polarity():
    amr_str = """(y2 / year~e.4
                      :time-of~e.5 (r / recover-01~e.7
                            :ARG1-of (e / expect-01 :polarity -~e.6))
                      :ARG1-of (p / possible-01~e.1)
                      :domain~e.2 (d / date-entity :year~e.4 2012~e.0))"""
    parsed_amr: AMR = AMR.parse_string(amr_str)
    expected_amr: AMR = AMR()
    expected_amr.node_to_concepts = {'y2': 'year',
                                     'r': 'recover-01',
                                     'e': 'expect-01',
                                     'p': 'possible-01',
                                     'd': 'date-entity'}
    expected_amr.node_to_tokens = {'y2': ['4'], 'r': ['7'], '-': [('6', 'e')], 'p': ['1'],'2012': [('0', 'd')]}
    expected_amr.relation_to_tokens = {'time-of': [('5', 'y2')], 'domain': [('2', 'y2')], 'year': [('4', 'd')]}
    expected_amr['y2'] = {'time-of': [('r',)], 'ARG1-of': [('p',)], 'domain': [('d',)]}
    expected_amr['r'] = {'ARG1-of': [('e',)]}
    expected_amr['e'] = {'polarity': [('-',)]}
    expected_amr['-'] = {}
    expected_amr['p'] = {}
    expected_amr['d'] = {'year': [('2012',)]}
    expected_amr['2012'] = {}
    assert_amr_graph_dictionaries(expected_amr, parsed_amr)


def test_parse_string():
    test_parse_example_1()
    test_parse_example_with_polarity()


if __name__ == "__main__":
    test_parse_string()
    print("Everything in amr_graph_test passed")
