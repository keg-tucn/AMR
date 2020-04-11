from models.amr_data import CustomizedAMR
from models.amr_graph import AMR


def assert_custom_amr_dictionaries(expected_amr: CustomizedAMR, parsed_amr: CustomizedAMR):
    assert expected_amr.tokens_to_concepts_dict == parsed_amr.tokens_to_concepts_dict, \
        'tokens_to_concepts_dict' + str(parsed_amr.tokens_to_concepts_dict) + \
        'should be' + str(expected_amr.tokens_to_concepts_dict)
    assert expected_amr.tokens_to_concept_list_dict == parsed_amr.tokens_to_concept_list_dict, \
        'tokens_to_concept_list_dict' + str(parsed_amr.tokens_to_concept_list_dict) + \
        'should be' + str(expected_amr.tokens_to_concept_list_dict)
    assert expected_amr.relations_dict == parsed_amr.relations_dict, \
        'relations_dict' + str(parsed_amr.relations_dict) + \
        'should be' + str(expected_amr.relations_dict)
    assert expected_amr.parent_dict == parsed_amr.parent_dict, \
        'parent_dict' + str(parsed_amr.parent_dict) + \
        'should be' + str(expected_amr.parent_dict)


# amr_str = """(r / recommend-01~e.1
#                 :ARG1 (a / advocate-01~e.4
#                     :ARG1 (i / it~e.0)
#                     :manner~e.2 (v / vigorous~e.3)))"""
# sentence = """It should be vigorously advocated ."""
def test_parse_example_1():
    amr: AMR = AMR()
    amr.node_to_concepts = {'i': 'it', 'v': 'vigorous', 'a': 'advocate-01', 'r': 'recommend-01'}
    amr.node_to_tokens = {'i': ['0'], 'v': ['3'], 'a': ['4'], 'r': ['1']}
    amr.relation_to_tokens = {'manner': [('2', 'a')]}
    amr['i'] = {}
    amr['v'] = {}
    amr['a'] = {'ARG1': [('i',)], 'manner': [('v',)]}
    amr['r'] = {'ARG1': [('a',)]}
    generated_custom_amr: CustomizedAMR = CustomizedAMR()
    generated_custom_amr.create_custom_AMR(amr)

    expected_custom_amr: CustomizedAMR = CustomizedAMR()
    expected_custom_amr.tokens_to_concepts_dict = {0: ('i', 'it'),
                                                   1: ('r', 'recommend-01'),
                                                   3: ('v', 'vigorous'),
                                                   4: ('a', 'advocate-01')}
    expected_custom_amr.tokens_to_concept_list_dict = {0: [('i', 'it')],
                                                       1: [('r', 'recommend-01')],
                                                       3: [('v', 'vigorous')],
                                                       4: [('a', 'advocate-01')]}
    # (child,parent) : (relation, children of child, token aligned to child)
    expected_custom_amr.relations_dict = {('i', 'a'): ('ARG1', [], ['0']),
                                          ('v', 'a'): ('manner', [], ['3']),
                                          ('r', ''): ('', ['a'], ['1']),
                                          ('a', 'r'): ('ARG1', ['i', 'v'], ['4'])}
    expected_custom_amr.parent_dict = {'i': 'a', 'v': 'a', 'a': 'r', 'r': ''}
    assert_custom_amr_dictionaries(expected_custom_amr, generated_custom_amr)


# ::id bolt12_10510_9791.5 ::amr-annotator SDL-AMR-09 ::preferred
# ::tok We should offer earthquake workers our full understanding .
# ::alignments 0-1.1.1 1-1 2-1.1 3-1.1.3.1.1 4-1.1.3 4-1.1.3.1 4-1.1.3.1.r 5-1.1.2.1 5-1.1.2.1.r 6-1.1.2.2 7-1.1.2
# (r / recommend-01~e.1
#       :ARG1 (o / offer-01~e.2
#             :ARG0 (w / we~e.0)
#             :ARG1 (u / understand-01~e.7
#                   :ARG0~e.5 w~e.5
#                   :mod (f / full~e.6))
#             :ARG3 (p / person~e.4
#                   :ARG0-of~e.4 (w2 / work-01~e.4
#                         :mod (e / earthquake~e.3)))))
def test_parse_example_2():
    amr: AMR = AMR()
    amr.node_to_concepts = {'r': 'recommend-01', 'o': 'offer-01', 'w': 'we', 'u': 'understand-01',
                            'f': 'full', 'p': 'person', 'w2': 'work-01', 'e': 'earthquake'}
    amr.node_to_tokens = {'r': ['1'], 'o': ['2'], 'w': ['5', '0'], 'u':['7'],'f': ['6'], 'p': ['4'], 'w2': ['4'], 'e': ['3']}
    amr.relation_to_tokens = {'ARG0': [('5', 'u')], 'ARG0-of': [('4', 'p')]}
    amr['r'] = {'ARG1': [('o',)]}
    amr['o'] = {'ARG0': [('w',)], 'ARG1': [('u',)], 'ARG3': [('p',)]}
    amr['w'] = {}
    amr['u'] = {'ARG0': [('w',)], 'mod': [('f',)]}
    amr['f'] = {}
    amr['p'] = {'ARG0-of': [('w2',)]}
    amr['w2'] = {'mod': [('e',)]}
    generated_custom_amr: CustomizedAMR = CustomizedAMR()
    generated_custom_amr.create_custom_AMR(amr)

    expected_custom_amr: CustomizedAMR = CustomizedAMR()
    expected_custom_amr.tokens_to_concepts_dict = {0: ('w', 'we'),
                                                   1: ('r', 'recommend-01'),
                                                   2: ('o', 'offer-01'),
                                                   3: ('e', 'earthquake'),
                                                   4: ('w2', 'work-01'),
                                                   5: ('w', 'we'),
                                                   6: ('f', 'full'),
                                                   7: ('u', 'understand-01')}
    expected_custom_amr.tokens_to_concept_list_dict = {0: [('w', 'we')],
                                                       1: [('r', 'recommend-01')],
                                                       2: [('o', 'offer-01')],
                                                       3: [('e', 'earthquake')],
                                                       4: [('p', 'person'), ('w2', 'work-01')],
                                                       5: [('w', 'we')],
                                                       6: [('f', 'full')],
                                                       7: [('u', 'understand-01')]}
    # (child,parent) : (relation, children of child, token aligned to child)
    expected_custom_amr.relations_dict = {('r', ''): ('', ['o'], ['1']),
                                          ('o', 'r'): ('ARG1', ['w', 'u', 'p'], ['2']),
                                          ('w', 'o'): ('ARG0', [], ['5','0']),
                                          ('u', 'o'): ('ARG1', ['w', 'f'], ['7']),
                                          ('p', 'o'): ('ARG3', ['w2'], ['4']),
                                          ('w', 'u'): ('ARG0', [], ['5','0']),
                                          ('f', 'u'): ('mod', [], ['6']),
                                          ('w2', 'p'): ('ARG0-of', ['e'], ['4']),
                                          ('e', 'w2'): ('mod', [], ['3'])}
    expected_custom_amr.parent_dict = {'r': '', 'o': 'r', 'w': 'o', 'u': 'o', 'p': 'o', 'w': 'u', 'f': 'u', 'w2': 'p',
                                       'e': 'w2'}
    assert_custom_amr_dictionaries(expected_custom_amr, generated_custom_amr)


def test_parse_string():
    test_parse_example_1()
    test_parse_example_2()


if __name__ == "__main__":
    test_parse_string()
    print("Everything in amr_data_test passed")
