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
    assert expected_amr.roots == parsed_amr.roots, 'Routs should match'


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
    expected_amr.roots = ['r']
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
    expected_amr.roots = ['y2']
    assert_amr_graph_dictionaries(expected_amr, parsed_amr)

# ::id bolt-eng-DF-170-181103-8889109_0085.26 ::amr-annotator UCO-AMR-05 ::preferred
# ::tok And the only reason they are there is because of insane student loan practices that are identical
# in everyway to the insane mortgage loan practices .
# amr_str = """(a / and~e.0
#                   :op2 (p2 / practice-01~e.13
#                         :ARG1 (l / loan-01~e.12
#                               :ARG2 (p / person~e.11
#                                     :ARG0-of~e.11 (s / study-01~e.11)))
#                         :mod (s2 / sane~e.10 :polarity~e.10 -~e.10)
#                         :ARG1-of (i2 / identical-01~e.16
#                               :ARG2~e.19 (p3 / practice-01~e.24
#                                     :ARG1 (l2 / loan-01~e.23
#                                           :ARG1 (m / mortgage-01~e.22))
#                                     :mod (s3 / sane~e.21 :polarity~e.21 -~e.21))
#                               :manner (w / way
#                                     :mod (e / every)))
#                         :ARG0-of (c2 / cause-01~e.3,8
#                               :ARG1 (b / be-located-at-91~e.5,7
#                                     :ARG1 (t / they~e.4)
#                                     :ARG2 (t2 / there~e.6))
#                               :mod (o / only~e.2))))"""
def test_parse_example_with_2_polarites():
    amr_str = """(a / and~e.0
                      :op2 (p2 / practice-01~e.13
                            :ARG1 (l / loan-01~e.12
                                  :ARG2 (p / person~e.11
                                        :ARG0-of~e.11 (s / study-01~e.11)))
                            :mod (s2 / sane~e.10 :polarity~e.10 -~e.10)
                            :ARG1-of (i2 / identical-01~e.16
                                  :ARG2~e.19 (p3 / practice-01~e.24
                                        :ARG1 (l2 / loan-01~e.23
                                              :ARG1 (m / mortgage-01~e.22))
                                        :mod (s3 / sane~e.21 :polarity~e.21 -~e.21))
                                  :manner (w / way
                                        :mod (e / every)))
                            :ARG0-of (c2 / cause-01~e.3,8 
                                  :ARG1 (b / be-located-at-91~e.5,7
                                        :ARG1 (t / they~e.4)
                                        :ARG2 (t2 / there~e.6))
                                  :mod (o / only~e.2))))"""
    parsed_amr: AMR = AMR.parse_string(amr_str)
    expected_amr: AMR = AMR()
    expected_amr.node_to_concepts = {'s': 'study-01',
                                     'p': 'person',
                                     'l': 'loan-01',
                                     's2': 'sane',
                                     'm': 'mortgage-01',
                                     'l2': 'loan-01',
                                     's3': 'sane',
                                     'p3': 'practice-01',
                                     'e': 'every',
                                     'w': 'way',
                                     'i2': 'identical-01',
                                     't': 'they',
                                     't2': 'there',
                                     'b': 'be-located-at-91',
                                     'o': 'only',
                                     'c2': 'cause-01',
                                     'p2': 'practice-01',
                                     'a': 'and'}
    expected_amr.node_to_tokens = {'s': ['11'], 'p': ['11'], '-': [('10', 's2'), ('21', 's3')], 'm': ['22'],
                                   'l2': ['23'], 's3': ['21'], 'p3': ['24'], 't': ['4'], 't2': ['6'],
                                   'b': ['5', '7'], 'o': ['2'], 'l': ['12'], 's2': ['10'], 'i2': ['16'],
                                   'c2': ['3', '8'], 'p2': ['13'], 'a': ['0']}

    expected_amr.relation_to_tokens = {'ARG0-of': [('11', 'p')], 'polarity': [('10', 's2'), ('21', 's3')], 'ARG2': [('19', 'i2')]}
    expected_amr['s'] = {}
    expected_amr['p'] = {'ARG0-of': [('s',)]}
    expected_amr['l'] = {'ARG2': [('p',)]}
    expected_amr['-'] = {}
    expected_amr['s2'] = {'polarity': [('-',)]}
    expected_amr['m'] = {}
    expected_amr['l2'] = {'ARG1': [('m',)]}
    expected_amr['s3'] = {'polarity': [('-',)]}
    expected_amr['p3'] = {'ARG1': [('l2',)], 'mod': [('s3',)]}
    expected_amr['e'] = {}
    expected_amr['w'] = {'mod': [('e',)]}
    expected_amr['i2'] = {'ARG2': [('p3',)], 'manner': [('w',)]}
    expected_amr['t'] = {}
    expected_amr['t2'] = {}
    expected_amr['b'] = {'ARG1': [('t',)], 'ARG2': [('t2',)]}
    expected_amr['o'] = {}
    expected_amr['c2'] = {'ARG1': [('b',)], 'mod': [('o',)]}
    expected_amr['p2'] = {'ARG1': [('l',)], 'mod': [('s2',)], 'ARG1-of': [('i2',)], 'ARG0-of': [('c2',)]}
    expected_amr['a'] = {'op2': [('p2',)]}
    expected_amr.roots = ['a']
    assert_amr_graph_dictionaries(expected_amr, parsed_amr)

# ::id bolt-eng-DF-170-181103-8889109_0077.18 ::amr-annotator UCO-AMR-05 ::preferred
# ::tok But the unemployment rep has the authority to apporve or denyt unemployment .
# amr_str = """(c / contrast-01~e.0
#                   :ARG2 (a2 / authorize-01~e.6
#                         :ARG1 (o2 / or~e.9
#                               :op1 (a / approve-01
#                                     :ARG0 p
#                                     :ARG1 (p2 / pay-01
#                                           :purpose (e2 / employ-01 :polarity -~e.2,11)))
#                               :op2 (d / deny-01
#                                     :ARG0 p
#                                     :ARG1 p2))
#                         :ARG2 (p / person
#                               :ARG0-of (r / represent-01
#                                     :ARG1 (o / organization
#                                           :mod (e / employ-01 :polarity -~e.2,11))))))"""
def test_parse_example2_with_2polarities():
    amr_str = """(c / contrast-01~e.0
                      :ARG2 (a2 / authorize-01~e.6
                            :ARG1 (o2 / or~e.9
                                  :op1 (a / approve-01
                                        :ARG0 p
                                        :ARG1 (p2 / pay-01
                                              :purpose (e2 / employ-01 :polarity -~e.2,11)))
                                  :op2 (d / deny-01
                                        :ARG0 p
                                        :ARG1 p2))
                            :ARG2 (p / person
                                  :ARG0-of (r / represent-01
                                        :ARG1 (o / organization
                                              :mod (e / employ-01 :polarity -~e.2,11))))))"""
    parsed_amr: AMR = AMR.parse_string(amr_str)
    expected_amr: AMR = AMR()
    expected_amr.node_to_concepts = {'e2': 'employ-01',
                                     'p2': 'pay-01',
                                     'a': 'approve-01',
                                     'd': 'deny-01',
                                     'o2': 'or',
                                     'e': 'employ-01',
                                     'o': 'organization',
                                     'r': 'represent-01',
                                     'p': 'person',
                                     'a2': 'authorize-01',
                                     'c': 'contrast-01'}

    expected_amr.node_to_tokens = {'-': [('2', 'e2'), ('11', 'e2'), ('2', 'e'), ('11', 'e')],
                                   'o2': ['9'], 'a2': ['6'], 'c': ['0']}

    expected_amr.relation_to_tokens = {}
    expected_amr['-'] = {}
    expected_amr['e2'] = {'polarity': [('-',)]}
    expected_amr['p2'] = {'purpose': [('e2',)]}
    expected_amr['p'] = {'ARG0-of': [('r',)]}
    expected_amr['a'] = {'ARG0': [('p',)], 'ARG1': [('p2',)]}
    expected_amr['d'] = {'ARG0': [('p',)], 'ARG1': [('p2',)]}
    expected_amr['o2'] = {'op1': [('a',)], 'op2': [('d',)]}
    expected_amr['e'] = {'polarity': [('-',)]}
    expected_amr['o'] = {'mod': [('e',)]}
    expected_amr['r'] = {'ARG1': [('o',)]}
    expected_amr['a2'] = {'ARG1': [('o2',)], 'ARG2': [('p',)]}
    expected_amr['c'] = {'ARG2': [('a2',)]}
    expected_amr.roots = ['c']
    assert_amr_graph_dictionaries(expected_amr, parsed_amr)


# ::id bolt12_632_5731.28 ::amr-annotator UCO-AMR-05 ::preferred
# ::tok We have now already received a payment reminder from the hospital .
# amr_str = """(r / receive-01~e.4
#                   :ARG0 (w / we~e.0)
#                   :ARG1 (t / thing~e.7
#                         :ARG0-of~e.7 (r2 / remind-01~e.7
#                               :ARG1 (p / pay-01~e.6
#                                     :ARG0 w)
#                               :ARG2 w))
#                   :ARG2~e.8 (h / hospital~e.10)
#                   :time (n / now~e.2)
#                   :time (a / already~e.3))"""
def test_parse_example_with_reentrancy():
    amr_str = """(r / receive-01~e.4
                      :ARG0 (w / we~e.0)
                      :ARG1 (t / thing~e.7
                            :ARG0-of~e.7 (r2 / remind-01~e.7
                                  :ARG1 (p / pay-01~e.6
                                        :ARG0 w)
                                  :ARG2 w))
                      :ARG2~e.8 (h / hospital~e.10)
                      :time (n / now~e.2)
                      :time (a / already~e.3))"""
    parsed_amr: AMR = AMR.parse_string(amr_str)
    expected_amr: AMR = AMR()
    expected_amr.node_to_concepts = {'w': 'we', 'p': 'pay-01', 'r2': 'remind-01',
                                     't': 'thing', 'h': 'hospital', 'n': 'now',
                                     'a': 'already', 'r': 'receive-01'}

    expected_amr.node_to_tokens = {'p': ['6'], 'r2': ['7'], 'w': ['0'], 't': ['7'],
                                   'h': ['10'], 'n': ['2'], 'a': ['3'], 'r': ['4']}

    expected_amr.relation_to_tokens = {'ARG0-of': [('7', 't')], 'ARG2': [('8', 'r')]}
    expected_amr['w'] = {}
    expected_amr['p'] = {'ARG0': [('w',)]}
    expected_amr['r2'] = {'ARG1': [('p',)], 'ARG2': [('w',)]}
    expected_amr['t'] = {'ARG0-of': [('r2',)]}
    expected_amr['h'] = {}
    expected_amr['n'] = {}
    expected_amr['a'] = {}
    expected_amr['r'] = {'ARG0': [('w',)], 'ARG1': [('t',)], 'ARG2': [('h',)], 'time': [('n',), ('a',)]}
    expected_amr.roots = ['r']
    assert_amr_graph_dictionaries(expected_amr, parsed_amr)

def test_parse_string():
    test_parse_example_1()
    test_parse_example_with_polarity()
    test_parse_example_with_2_polarites()
    test_parse_example2_with_2polarities()
    test_parse_example_with_reentrancy()

if __name__ == "__main__":
    test_parse_string()
    print("Everything in amr_graph_test passed")
