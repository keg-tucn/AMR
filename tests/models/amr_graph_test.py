from models.amr_graph import AMR


def assert_amr_graph_dictionaries(expected_amr: AMR, parsed_amr: AMR):
    assert expected_amr.node_to_concepts == parsed_amr.node_to_concepts, \
        'Node to concepts' + str(parsed_amr.node_to_concepts) + 'should be' + str(expected_amr.node_to_concepts)
    assert expected_amr.node_to_tokens == parsed_amr.node_to_tokens,\
        'Node to tokens' +  str(parsed_amr.node_to_tokens) + 'should be' + str(expected_amr.node_to_tokens)
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


def test_parse_string():
    test_parse_example_1()


if __name__ == "__main__":
    test_parse_string()
    print("Everything in amr_graph_test passed")
