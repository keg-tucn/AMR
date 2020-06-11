from models.amr_graph import AMR
from pre_post_processing.standford_pre_post_processing import train_pre_processing


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


# sentence = Comrade Deng Xiaoping once said that the Communist Party will not be overthrown - if it falls ,
# it will be brought down from within the party itself .
# amr_str = """(s / say-01~e.4
#   :ARG0 (p / person :wiki "Deng_Xiaoping"
#         :name~e.1 (n / name :op1 "Deng"~e.1 :op2 "Xiaoping"~e.2)
#         :ARG0-of (h / have-rel-role-91
#               :ARG2 (c / comrade~e.0)))
#   :ARG1~e.5 (a / and
#         :op1 (o2 / overthrow-01~e.12 :polarity~e.10 -~e.10
#               :ARG1 (p2 / political-party~e.26 :wiki "Communist_Party_of_China"
#                     :name (n2 / name :op1 "Communist"~e.7 :op2 "Party"~e.8)))
#         :op2 (b / bring-down-03~e.21,22
#               :ARG0 (p3 / person
#                     :ARG0-of (h2 / have-org-role-91
#                           :ARG1 p2))
#               :ARG1 p2~e.18
#               :condition~e.14 (f / fall-05~e.16
#                     :ARG1 p2~e.15)))
#   :time (o / once~e.3))"""
def test_train_pre_processing_ex_person():
    amr = AMR()
    amr.roots = ['s']
    amr.reentrance_triples = [('h2', 'ARG1', 'p2'), ('f', 'ARG1', 'p2'), ('b', 'ARG1', 'p2')]
    amr.node_to_concepts = {'n': 'name', 'c': 'comrade', 'h': 'have-rel-role-91', 'p': 'person', 'n2': 'name',
                            'p2': 'political-party', 'o2': 'overthrow-01', 'h2': 'have-org-role-91', 'p3': 'person',
                            'f': 'fall-05', 'b': 'bring-down-03', 'a': 'and', 'o': 'once', 's': 'say-01'}
    amr.node_to_tokens = {'Deng': [('1', 'n')], 'Xiaoping': [('2', 'n')],
                          'c': ['0'], 'Communist': [('7', 'n2')], 'Party': [('8', 'n2')],
                          '-': [('10', 'o2')], 'p2': ['26', '15', '18'], 'f': ['16'],
                          'o2': ['12'], 'b': ['21', '22'], 'o': ['3'], 's': ['4']}
    amr.relation_to_tokens = {'polarity': [('10', 'o2')], 'condition': [('14', 'b')], 'ARG1': [('5', 's')],
                              'name': [('1', 'p')]}
    amr['Deng'] = {}
    amr['n'] = {'op1': [('Deng',)], 'op2': [('Xiaoping',)]}
    amr['Xiaoping'] = {}
    amr['c'] = {}
    amr['h'] = {'ARG2': [('c',)]}
    amr['Deng_Xiaoping'] = {}
    amr['p'] = {'wiki': [('Deng_Xiaoping',)], 'name': [('n',)], 'ARG0-of': [('h',)]}
    amr['Communist'] = {}
    amr['n2'] = {'op1': [('Communist',)], 'op2': [('Party',)]}
    amr['Party'] = {}
    amr['Communist_Party_of_China'] = {}
    amr['p2'] = {'wiki': [('Communist_Party_of_China',)], 'name': [('n2',)]}
    amr['-'] = {}
    amr['o2'] = {'polarity': [('-',)], 'ARG1': [('p2',)]}
    amr['h2'] = {'ARG1': [('p2',)]}
    amr['p3'] = {'ARG0-of': [('h2',)]}
    amr['f'] = {'ARG1': [('p2',)]}
    amr['b'] = {'ARG0': [('p3',)], 'ARG1': [('p2',)], 'condition': [('f',)]}
    amr['a'] = {'op1': [('o2',)], 'op2': [('b',)]}
    amr['o'] = {}
    amr['s'] = {'ARG0': [('p',)], 'ARG1': [('a',)], 'time': [('o',)]}
    sentence = 'Comrade Deng Xiaoping once said that the Communist Party will not be overthrown - ' \
               'if it falls , it will be brought down from within the party itself .'
    generated_amr, generated_sentence = train_pre_processing(amr, sentence)
    expected_sentence = 'Comrade PERSON once said that the Communist Party will not be overthrown - ' \
                        'if it falls , it will be brought down from within the party itself .'
    expected_amr = AMR()
    expected_amr.roots = ['s']
    expected_amr.reentrance_triples = [('h2', 'ARG1', 'p2'), ('f', 'ARG1', 'p2'), ('b', 'ARG1', 'p2')]
    expected_amr.node_to_concepts = {'c': 'comrade', 'h': 'have-rel-role-91', 'p': 'PERSON', 'n2': 'name',
                                     'p2': 'political-party', 'o2': 'overthrow-01', 'h2': 'have-org-role-91',
                                     'p3': 'person',
                                     'f': 'fall-05', 'b': 'bring-down-03', 'a': 'and', 'o': 'once', 's': 'say-01'}
    expected_amr.node_to_tokens = {'p': ['1'],
                                   'c': ['0'], 'Communist': [('6', 'n2')], 'Party': [('7', 'n2')],
                                   '-': [('9', 'o2')], 'p2': ['25', '14', '17'], 'f': ['15'],
                                   'o2': ['11'], 'b': ['20', '21'], 'o': ['2'], 's': ['3']}
    expected_amr.relation_to_tokens = {'polarity': [('9', 'o2')], 'condition': [('13', 'b')], 'ARG1': [('4', 's')]}
    expected_amr['c'] = {}
    expected_amr['h'] = {'ARG2': [('c',)]}
    expected_amr['p'] = {'ARG0-of': [('h',)]}
    expected_amr['Communist'] = {}
    expected_amr['n2'] = {'op1': [('Communist',)], 'op2': [('Party',)]}
    expected_amr['Party'] = {}
    expected_amr['Communist_Party_of_China'] = {}
    expected_amr['p2'] = {'wiki': [('Communist_Party_of_China',)], 'name': [('n2',)]}
    expected_amr['-'] = {}
    expected_amr['o2'] = {'polarity': [('-',)], 'ARG1': [('p2',)]}
    expected_amr['h2'] = {'ARG1': [('p2',)]}
    expected_amr['p3'] = {'ARG0-of': [('h2',)]}
    expected_amr['f'] = {'ARG1': [('p2',)]}
    expected_amr['b'] = {'ARG0': [('p3',)], 'ARG1': [('p2',)], 'condition': [('f',)]}
    expected_amr['a'] = {'op1': [('o2',)], 'op2': [('b',)]}
    expected_amr['o'] = {}
    expected_amr['s'] = {'ARG0': [('p',)], 'ARG1': [('a',)], 'time': [('o',)]}
    assert_amr_graph_dictionaries(expected_amr, generated_amr)
    assert generated_sentence == expected_sentence


# TODO: see if it makes sense investing in differentiating negatives and no wiki entries (both literal: - )
# ::tok Some propaganda activities of ZF have soon become viewed as jokes by the people .
# (b / become-01~e.7
#       :ARG1 a
#       :ARG2 (v / view-02~e.8
#             :ARG0~e.11 (p2 / person~e.13)
#             :ARG1 (a / activity-06~e.2
#                   :ARG0~e.3 (o / organization :wiki -
#                         :name (n / name :op1 "ZF"~e.4))
#                   :ARG1 (p / propaganda~e.1)
#                   :quant (s / some~e.0))
#             :ARG2~e.9 (t / thing~e.10
#                   :ARG2-of~e.10 (j / joke-01~e.10)))
#       :time~e.9 (s2 / soon~e.6))
def test_train_pre_processing_ex_organization():
    sentence = "Some propaganda activities of ZF have soon become viewed as jokes by the people ."
    amr: AMR = AMR()
    amr.roots = ['b']
    amr.reentrance_triples = [('b', 'ARG1', 'a')]
    amr.node_to_concepts = {'p2': 'person', 'n': 'name', 'o': 'organization', 'p': 'propaganda',
                            's': 'some', 'a': 'activity-06', 'j': 'joke-01', 't': 'thing',
                            'v': 'view-02', 's2': 'soon', 'b': 'become-01'}
    amr.node_to_tokens = {'ZF': [('4', 'n')], 'p': ['1'], 's': ['0'], 'j': ['10'],
                          'p2': ['13'], 'a': ['2'], 't': ['10'], 'v': ['8'], 's2': ['6'], 'b': ['7']}
    amr.relation_to_tokens = {'ARG0': [('3', 'a'), ('11', 'v')], 'ARG2-of': [('10', 't')], 'ARG2': [('9', 'v')],
                              'time': [('9', 'b')]}
    amr['p2'] = {}
    amr['ZF'] = {}
    amr['n'] = {'op1': [('ZF',)]}
    amr['-'] = {}
    amr['o'] = {'wiki': [('-',)], 'name': [('n',)]}
    amr['p'] = {}
    amr['s'] = {}
    amr['a'] = {'ARG0': [('o',)], 'ARG1': [('p',)], 'quant': [('s',)]}
    amr['j'] = {}
    amr['t'] = {'ARG2-of': [('j',)]}
    amr['v'] = {'ARG0': [('p2',)], 'ARG1': [('a',)], 'ARG2': [('t',)]}
    amr['s2'] = {}
    amr['b'] = {'ARG1': [('a',)], 'ARG2': [('v',)], 'time': [('s2',)]}
    generated_amr, generated_sentence = train_pre_processing(amr, sentence)
    # Expected
    expected_sentence = "Some propaganda activities of ORGANIZATION have soon become viewed as jokes by the people ."
    expected_amr: AMR = AMR()
    expected_amr.roots = ['b']
    expected_amr.reentrance_triples = [('b', 'ARG1', 'a')]
    expected_amr.node_to_concepts = {'p2': 'person', 'o': 'ORGANIZATION', 'p': 'propaganda',
                                     's': 'some', 'a': 'activity-06', 'j': 'joke-01', 't': 'thing',
                                     'v': 'view-02', 's2': 'soon', 'b': 'become-01'}
    expected_amr.node_to_tokens = {'o': ['4'], 'p': ['1'], 's': ['0'], 'j': ['10'],
                                   'p2': ['13'], 'a': ['2'], 't': ['10'], 'v': ['8'], 's2': ['6'], 'b': ['7']}
    expected_amr.relation_to_tokens = {'ARG0': [('3', 'a'), ('11', 'v')], 'ARG2-of': [('10', 't')],
                                       'ARG2': [('9', 'v')],
                                       'time': [('9', 'b')]}
    expected_amr['p2'] = {}
    expected_amr['o'] = {}
    expected_amr['p'] = {}
    expected_amr['s'] = {}
    expected_amr['a'] = {'ARG0': [('o',)], 'ARG1': [('p',)], 'quant': [('s',)]}
    expected_amr['j'] = {}
    expected_amr['t'] = {'ARG2-of': [('j',)]}
    expected_amr['v'] = {'ARG0': [('p2',)], 'ARG1': [('a',)], 'ARG2': [('t',)]}
    expected_amr['s2'] = {}
    expected_amr['b'] = {'ARG1': [('a',)], 'ARG2': [('v',)], 'time': [('s2',)]}
    assert_amr_graph_dictionaries(expected_amr, generated_amr)
    assert generated_sentence == expected_sentence


# ::id bolt12_6454_5051.12 ::amr-annotator SDL-AMR-09 ::preferred
# ::tok the essay " Are you studying Paul or studying Bill ? ", she blathers that " people who study Paul will become terrorists ";
# (b2 / blather-01~e.13
#       :ARG0 (s2 / she~e.12)
#       :ARG1 (b / become-01~e.21
#             :ARG1 (p2 / person~e.16
#                   :ARG0-of (s3 / study-01~e.18
#                         :ARG1 (p3 / person :wiki -
#                               :name (n2 / name :op1 "Paul"~e.19))))
#             :ARG2 (t2 / terrorist~e.22))
#       :medium (e / essay~e.1
#             :ARG1-of (t / title-01
#                   :ARG2 (s / study-01~e.5,8 :mode~e.10 interrogative~e.10
#                         :ARG0 (y / you~e.4)
#                         :ARG1 (o / or~e.7
#                               :op1 p3~e.6
#                               :op2 (p4 / person :wiki -
#                                     :name (n / name :op1 "Bill"~e.9)))))))
def test_train_pre_processing_ex_2_person():
    sentence = 'the essay " Are you studying Paul or studying Bill ? ", she blathers that " people who study Paul ' \
               'will become terrorists "; '
    amr: AMR = AMR()
    amr.roots = ['b2']
    amr.reentrance_triples = [('o', 'op1', 'p3')]
    amr.node_to_concepts = {'s2': 'she', 'n2': 'name', 'p3': 'person', 's3': 'study-01', 'p2': 'person',
                            't2': 'terrorist', 'b': 'become-01', 'y': 'you', 'n': 'name', 'p4': 'person',
                            'o': 'or', 's': 'study-01', 't': 'title-01', 'e': 'essay', 'b2': 'blather-01'}
    amr.node_to_tokens = {'Paul': [('19', 'n2')], 's3': ['18'], 'p2': ['16'], 't2': ['22'],
                          'Bill': [('9', 'n')], 'p3': ['6'], 'interrogative': [('10', 's')], 'y': ['4'],
                          'o': ['7'], 's': ['5', '8'], 's2': ['12'], 'b': ['21'], 'e': ['1'], 'b2': ['13']}
    amr.relation_to_tokens = {'mode': [('10', 's')]}
    amr['s2'] = {}
    amr['Paul'] = {}
    amr['n2'] = {'op1': [('Paul',)]}
    amr['-'] = {}
    amr['p3'] = {'wiki': [('-',)], 'name': [('n2',)]}
    amr['s3'] = {'ARG1': [('p3',)]}
    amr['p2'] = {'ARG0-of': [('s3',)]}
    amr['t2'] = {}
    amr['b'] = {'ARG1': [('p2',)], 'ARG2': [('t2',)]}
    amr['y'] = {}
    amr['Bill'] = {}
    amr['n'] = {'op1': [('Bill',)]}
    amr['p4'] = {'wiki': [('-',)], 'name': [('n',)]}
    amr['o'] = {'op1': [('p3',)], 'op2': [('p4',)]}
    amr['interrogative'] = {}
    amr['s'] = {'mode': [('interrogative',)], 'ARG0': [('y',)], 'ARG1': [('o',)]}
    amr['t'] = {'ARG2': [('s',)]}
    amr['e'] = {'ARG1-of': [('t',)]}
    amr['b2'] = {'ARG0': [('s2',)], 'ARG1': [('b',)], 'medium': [('e',)]}
    generated_amr, generated_sentence = train_pre_processing(amr, sentence)

    # Expected
    expected_sentence = 'the essay " Are you studying PERSON or studying PERSON ? ", she blathers that " people who study PERSON ' \
                        'will become terrorists ";'
    expected_amr: AMR = AMR()
    expected_amr.roots = ['b2']
    expected_amr.reentrance_triples = [('o', 'op1', 'p3')]
    expected_amr.node_to_concepts = {'s2': 'she', 'p3': 'PERSON', 's3': 'study-01', 'p2': 'person',
                                     't2': 'terrorist', 'b': 'become-01', 'y': 'you', 'p4': 'PERSON',
                                     'o': 'or', 's': 'study-01', 't': 'title-01', 'e': 'essay', 'b2': 'blather-01'}
    expected_amr.node_to_tokens = {'s3': ['18'], 'p2': ['16'], 't2': ['22'],
                                   'p4': ['9'], 'p3': ['6'], 'interrogative': [('10', 's')], 'y': ['4'],
                                   'o': ['7'], 's': ['5', '8'], 's2': ['12'], 'b': ['21'], 'e': ['1'], 'b2': ['13']}
    expected_amr.relation_to_tokens = {'mode': [('10', 's')]}
    expected_amr['s2'] = {}
    expected_amr['p3'] = {}
    expected_amr['s3'] = {'ARG1': [('p3',)]}
    expected_amr['p2'] = {'ARG0-of': [('s3',)]}
    expected_amr['t2'] = {}
    expected_amr['b'] = {'ARG1': [('p2',)], 'ARG2': [('t2',)]}
    expected_amr['y'] = {}
    expected_amr['p4'] = {}
    expected_amr['o'] = {'op1': [('p3',)], 'op2': [('p4',)]}
    expected_amr['interrogative'] = {}
    expected_amr['s'] = {'mode': [('interrogative',)], 'ARG0': [('y',)], 'ARG1': [('o',)]}
    expected_amr['t'] = {'ARG2': [('s',)]}
    expected_amr['e'] = {'ARG1-of': [('t',)]}
    expected_amr['b2'] = {'ARG0': [('s2',)], 'ARG1': [('b',)], 'medium': [('e',)]}
    assert_amr_graph_dictionaries(expected_amr, generated_amr)
    assert generated_sentence == expected_sentence


# ::id bolt12_10494_3592.5 ::amr-annotator SDL-AMR-09 ::preferred
# ::tok Now , Wang Shi said , these responses have had effects on me .
# (s / say-01~e.4
#       :ARG0 (p / person :wiki "Wang_Shi_(entrepreneur)"
#             :name (n2 / name :op1 "Wang"~e.2 :op2 "Shi"~e.3))
#       :ARG1 (e / effect-03~e.10
#             :ARG0 (t2 / thing~e.7
#                   :ARG2-of~e.7 (r / respond-01~e.7)
#                   :mod (t / this~e.6))
#             :ARG1~e.11 p~e.12)
#       :time (n / now~e.0))
def test_train_pre_processing_ex_person_reentrancy():
    sentence = 'Now , Wang Shi said , these responses have had effects on me .'
    amr: AMR = AMR()
    amr.roots = ['s']
    amr.reentrance_triples = [('e', 'ARG1', 'p')]
    amr.node_to_concepts = {'n2': 'name', 'p': 'person', 'r': 'respond-01', 't': 'this',
                            't2': 'thing', 'e': 'effect-03', 'n': 'now', 's': 'say-01'}
    amr.node_to_tokens = {'Wang': [('2', 'n2')], 'Shi': [('3', 'n2')], 'r': ['7'], 't': ['6'],
                          't2': ['7'], 'p': ['12'], 'e': ['10'], 'n': ['0'], 's': ['4']}
    amr.relation_to_tokens = {'ARG2-of': [('7', 't2')], 'ARG1': [('11', 'e')]}
    amr['Wang'] = {}
    amr['n2'] = {'op1': [('Wang',)], 'op2': [('Shi',)]}
    amr['Shi'] = {}
    amr['Wang_Shi_(entrepreneur)'] = {}
    amr['p'] = {'wiki': [('Wang_Shi_(entrepreneur)',)], 'name': [('n2',)]}
    amr['r'] = {}
    amr['t'] = {}
    amr['t2'] = {'ARG2-of': [('r',)], 'mod': [('t',)]}
    amr['e'] = {'ARG0': [('t2',)], 'ARG1': [('p',)]}
    amr['n'] = {}
    amr['s'] = {'ARG0': [('p',)], 'ARG1': [('e',)], 'time': [('n',)]}
    generated_amr, generated_sentence = train_pre_processing(amr, sentence)

    # expected
    expected_sentence = 'Now , PERSON said , these responses have had effects on me .'
    expected_amr: AMR = AMR()
    expected_amr.roots = ['s']
    expected_amr.reentrance_triples = [('e', 'ARG1', 'p')]
    expected_amr.node_to_concepts = {'p': 'PERSON', 'r': 'respond-01', 't': 'this',
                                     't2': 'thing', 'e': 'effect-03', 'n': 'now', 's': 'say-01'}
    expected_amr.node_to_tokens = {'p': ['2'], 'r': ['6'], 't': ['5'],
                                   't2': ['6'], 'p': ['11'], 'e': ['9'], 'n': ['0'], 's': ['3']}
    expected_amr.relation_to_tokens = {'ARG2-of': [('6', 't2')], 'ARG1': [('10', 'e')]}
    expected_amr['p'] = {}
    expected_amr['r'] = {}
    expected_amr['t'] = {}
    expected_amr['t2'] = {'ARG2-of': [('r',)], 'mod': [('t',)]}
    expected_amr['e'] = {'ARG0': [('t2',)], 'ARG1': [('p',)]}
    expected_amr['n'] = {}
    expected_amr['s'] = {'ARG0': [('p',)], 'ARG1': [('e',)], 'time': [('n',)]}

    assert_amr_graph_dictionaries(expected_amr, generated_amr)
    assert generated_sentence == expected_sentence


# ::id bolt12_10510_9581.4 ::amr-annotator SDL-AMR-09 ::preferred
# ::tok But we can see such philanthropists as Bill Gates , Li Ka @-@ shing , Jackie Chan have not affected the growth of their enterprises in any way . Instead , they have actually increased the competitiveness of their companies .
# (m / multi-sentence
#       :snt1 (c / contrast-01~e.0
#             :ARG2 (p / possible-01~e.2
#                   :ARG1 (s / see-01~e.3
#                         :ARG0 (w / we~e.1)
#                         :ARG1 (a / affect-01~e.19 :polarity~e.18 -~e.18
#                               :ARG0 (p2 / philanthropist~e.5
#                                     :example~e.6 (a3 / and
#                                           :op1 (p3 / person :wiki "Bill_Gates"
#                                                 :name (n / name :op1 "Bill"~e.7 :op2 "Gates"~e.8))
#                                           :op2 (p4 / person :wiki "Li_Ka-shing"
#                                                 :name (n2 / name :op1 "Li"~e.10 :op2 "Ka-shing"~e.11,13))
#                                           :op3 (p5 / person :wiki "Jackie_Chan"
#                                                 :name (n3 / name :op1 "Jackie"~e.15 :op2 "Chan"~e.16))))
#                               :ARG1 (g / grow-01~e.21
#                                     :ARG1 (e / enterprise~e.24
#                                           :poss p2))
#                               :manner~e.25 (w2 / way~e.27
#                                     :mod (a2 / any~e.26))))))
#       :snt2 (i / increase-01~e.34
#             :ARG0 (t / they~e.23,31)
#             :ARG1 (c2 / competitiveness~e.36
#                   :poss~e.37 (c3 / company~e.39
#                         :poss~e.38 t~e.38))
#             :ARG1-of (a4 / actual-02~e.33)
#             :ARG1-of (i2 / instead-of-91~e.22,29,37)))
def test_train_pre_processing_ex_3_person():
    sentence = 'But we can see such philanthropists as Bill Gates , Li Ka @-@ shing , Jackie Chan have not affected ' \
               'the growth of their enterprises in any way . Instead , they have actually increased the ' \
               'competitiveness of their companies . '

    amr: AMR = AMR()
    amr.roots = ['m']
    amr.reentrance_triples = [('e', 'poss', 'p2'), ('c3', 'poss', 't')]
    amr.node_to_concepts = {'w': 'we', 'n': 'name', 'p3': 'person', 'n2': 'name', 'p4': 'person',
                            'n3': 'name', 'p5': 'person', 'a3': 'and', 'p2': 'philanthropist',
                            'e': 'enterprise', 'g': 'grow-01', 'a2': 'any', 'w2': 'way', 'a': 'affect-01',
                            's': 'see-01', 'p': 'possible-01', 'c': 'contrast-01', 't': 'they', 'c3': 'company',
                            'c2': 'competitiveness', 'a4': 'actual-02', 'i2': 'instead-of-91', 'i': 'increase-01',
                            'm': 'multi-sentence'}
    amr.node_to_tokens = {'Bill': [('7', 'n')], 'Gates': [('8', 'n')], 'Li': [('10', 'n2')],
                          'Ka-shing': [('11', 'n2'), ('13', 'n2')],
                          'Jackie': [('15', 'n3')], 'Chan': [('16', 'n3')], 'e': ['24'], 'a2': ['26'],
                          '-': [('18', 'a')],
                          'p2': ['5'], 'g': ['21'], 'w2': ['27'], 'w': ['1'], 'a': ['19'], 's': ['3'],
                          'p': ['2'],
                          't': ['38', '23', '31'], 'c3': ['39'], 'c2': ['36'], 'a4': ['33'],
                          'i2': ['22', '29', '37'],
                          'c': ['0'], 'i': ['34']}
    amr.relation_to_tokens = {'example': [('6', 'p2')], 'polarity': [('18', 'a')], 'manner': [('25', 'a')],
                              'poss': [('38', 'c3'), ('37', 'c2')]}
    amr['w'] = {}
    amr['Bill'] = {}
    amr['n'] = {'op1': [('Bill',)], 'op2': [('Gates',)]}
    amr['Gates'] = {}
    amr['Bill_Gates'] = {}
    amr['p3'] = {'wiki': [('Bill_Gates',)], 'name': [('n',)]}
    amr['Li'] = {}
    amr['n2'] = {'op1': [('Li',)], 'op2': [('Ka-shing',)]}
    amr['Ka-shing'] = {}
    amr['Li_Ka-shing'] = {}
    amr['p4'] = {'wiki': [('Li_Ka-shing',)], 'name': [('n2',)]}
    amr['Jackie'] = {}
    amr['n3'] = {'op1': [('Jackie',)], 'op2': [('Chan',)]}
    amr['Chan'] = {}
    amr['Jackie_Chan'] = {}
    amr['p5'] = {'wiki': [('Jackie_Chan',)], 'name': [('n3',)]}
    amr['a3'] = {'op1': [('p3',)], 'op2': [('p4',)], 'op3': [('p5',)]}
    amr['p2'] = {'example': [('a3',)]}
    amr['e'] = {'poss': [('p2',)]}
    amr['g'] = {'ARG1': [('e',)]}
    amr['a2'] = {}
    amr['w2'] = {'mod': [('a2',)]}
    amr['-'] = {}
    amr['a'] = {'polarity': [('-',)], 'ARG0': [('p2',)], 'ARG1': [('g',)], 'manner': [('w2',)]}
    amr['s'] = {'ARG0': [('w',)], 'ARG1': [('a',)]}
    amr['p'] = {'ARG1': [('s',)]}
    amr['c'] = {'ARG2': [('p',)]}
    amr['t'] = {}
    amr['c3'] = {'poss': [('t',)]}
    amr['c2'] = {'poss': [('c3',)]}
    amr['a4'] = {'poss': [('c3',)]}
    amr['i2'] = {}
    amr['i'] = {'ARG0': [('t',)], 'ARG1': [('c2',)], 'ARG1-of': [('a4',), ('i2',)]}
    amr['m'] = {'snt1': [('c',)], 'snt2': [('i',)]}
    generated_amr, generated_sentence = train_pre_processing(amr, sentence)

    # expected
    expected_sentence = 'But we can see such philanthropists as PERSON , PERSON Ka @-@ shing , PERSON have not affected ' \
                        'the growth of their enterprises in any way . Instead , they have actually increased the ' \
                        'competitiveness of their companies .'
    expected_amr: AMR = AMR()
    expected_amr.roots = ['m']
    expected_amr.reentrance_triples = [('e', 'poss', 'p2'), ('c3', 'poss', 't')]
    expected_amr.node_to_concepts = {'w': 'we', 'p3': 'PERSON', 'p4': 'PERSON',
                                     'p5': 'PERSON', 'a3': 'and', 'p2': 'philanthropist',
                                     'e': 'enterprise', 'g': 'grow-01', 'a2': 'any', 'w2': 'way', 'a': 'affect-01',
                                     's': 'see-01', 'p': 'possible-01', 'c': 'contrast-01', 't': 'they',
                                     'c3': 'company',
                                     'c2': 'competitiveness', 'a4': 'actual-02', 'i2': 'instead-of-91',
                                     'i': 'increase-01',
                                     'm': 'multi-sentence'}
    expected_amr.node_to_tokens = {'p3': ['7'], 'p4': ['9'],
                                   'p5': ['14'], 'e': ['22'], 'a2': ['24'],
                                   '-': [('16', 'a')],
                                   'p2': ['5'], 'g': ['19'], 'w2': ['25'], 'w': ['1'], 'a': ['17'], 's': ['3'],
                                   'p': ['2'],
                                   't': ['36', '21', '29'], 'c3': ['37'], 'c2': ['34'], 'a4': ['31'],
                                   'i2': ['20', '27', '35'],
                                   'c': ['0'], 'i': ['32']}
    expected_amr.relation_to_tokens = {'example': [('6', 'p2')], 'polarity': [('16', 'a')], 'manner': [('23', 'a')],
                                       'poss': [('36', 'c3'), ('35', 'c2')]}
    expected_amr['w'] = {}
    expected_amr['p3'] = {}
    expected_amr['p4'] = {}
    expected_amr['p5'] = {}
    expected_amr['a3'] = {'op1': [('p3',)], 'op2': [('p4',)], 'op3': [('p5',)]}
    expected_amr['p2'] = {'example': [('a3',)]}
    expected_amr['e'] = {'poss': [('p2',)]}
    expected_amr['g'] = {'ARG1': [('e',)]}
    expected_amr['a2'] = {}
    expected_amr['w2'] = {'mod': [('a2',)]}
    expected_amr['-'] = {}
    expected_amr['a'] = {'polarity': [('-',)], 'ARG0': [('p2',)], 'ARG1': [('g',)], 'manner': [('w2',)]}
    expected_amr['s'] = {'ARG0': [('w',)], 'ARG1': [('a',)]}
    expected_amr['p'] = {'ARG1': [('s',)]}
    expected_amr['c'] = {'ARG2': [('p',)]}
    expected_amr['t'] = {}
    expected_amr['c3'] = {'poss': [('t',)]}
    expected_amr['c2'] = {'poss': [('c3',)]}
    expected_amr['a4'] = {'poss': [('c3',)]}
    expected_amr['i2'] = {}
    expected_amr['i'] = {'ARG0': [('t',)], 'ARG1': [('c2',)], 'ARG1-of': [('a4',), ('i2',)]}
    expected_amr['m'] = {'snt1': [('c',)], 'snt2': [('i',)]}
    assert generated_sentence == expected_sentence
    assert_amr_graph_dictionaries(expected_amr, generated_amr)


# ::id bolt12_10511_2891.4 ::amr-annotator SDL-AMR-09 ::preferred
# ::tok On the day of the Tangshan Earthquake , i.e. July 28 th , those on duty at Mao Zedong 's quarters were Wang Dongxing , Wang Hongwen , and Mao Zedong 's confidential secretary , Zhang Yufeng .
# (p / person
#       :prep-on~e.0,14 (d / duty~e.15
#             :time (d2 / day~e.2
#                   :time-of~e.3 (e / earthquake :wiki "1976_Tangshan_earthquake"
#                         :name (n5 / name :op1 "Tangshan"~e.5 :op2 "Earthquake"~e.6))
#                   :ARG1-of (m / mean-01
#                         :ARG2 (d3 / date-entity :month~e.9 7~e.9 :day 28~e.10)))
#             :location~e.16 (q / quarter~e.20
#                   :poss~e.19 p5~e.17,18))
#       :domain~e.8,19,21 (a / and~e.28
#             :op1 (p2 / person :wiki "Wang_Dongxing"
#                   :name (n / name :op1 "Wang"~e.22 :op2 "Dongxing"~e.23))
#             :op2 (p3 / person :wiki "Wang_Hongwen"
#                   :name (n2 / name :op1 "Wang"~e.25 :op2 "Hongwen"~e.26))
#             :op3 (p4 / person :wiki -
#                   :name (n3 / name :op1 "Zhang"~e.35 :op2 "Yufeng"~e.36)
#                   :ARG0-of (h / have-org-role-91~e.31
#                         :ARG1 (p5 / person :wiki "Mao_Zedong"
#                               :name (n4 / name :op1 "Mao"~e.29 :op2 "Zedong"~e.30))
#                         :ARG2 (s / secretary~e.33
#                               :mod (c / confidence))))))
def train_pre_processing_ex_persons_with_common_tokens():
    sentence = 'On the day of the Tangshan Earthquake , i.e. July 28 th , those on duty at Mao Zedong \'s quarters ' \
               'were Wang Dongxing , Wang Hongwen , and Mao Zedong \'s confidential secretary , Zhang Yufeng . '
    amr: AMR = AMR()
    amr.roots = ['p']
    amr.reentrance_triples = [('q', 'poss', 'p5')]
    amr.node_to_concepts = {'n5': 'name', 'e': 'earthquake', 'd3': 'date-entity', 'm': 'mean-01',
                            'd2': 'day', 'q': 'quarter', 'd': 'duty', 'n': 'name', 'p2': 'person',
                            'n2': 'name', 'p3': 'person', 'n3': 'name', 'n4': 'name', 'p5': 'person',
                            'c': 'confidence', 's': 'secretary', 'h': 'have-org-role-91', 'p4': 'person',
                            'a': 'and', 'p': 'person'}
    amr.node_to_tokens = {'Tangshan': [('5', 'n5')], 'Earthquake': [('6', 'n5')], '7': [('9', 'd3')],
                          '28': [('10', 'd3')], 'p5': ['17', '18'], 'd2': ['2'], 'q': ['20'],
                          'Wang': [('22', 'n'), ('25', 'n2')], 'Dongxing': [('23', 'n')],
                          'Hongwen': [('26', 'n2')], 'Zhang': [('35', 'n3')],
                          'Yufeng': [('36', 'n3')], 'Mao': [('29', 'n4')], 'Zedong': [('30', 'n4')],
                          's': ['33'], 'h': ['31'], 'd': ['15'], 'a': ['28']}
    amr.relation_to_tokens = {'month': [('9', 'd3')], 'time-of': [('3', 'd2')], 'poss': [('19', 'q')],
                              'location': [('16', 'd')], 'prep-on': [('0', 'p'), ('14', 'p')],
                              'domain': [('8', 'p'), ('19', 'p'), ('21', 'p')]}
    amr['Tangshan'] = {}
    amr['n5'] = {'op1': [('Tangshan',)], 'op2': [('Earthquake',)]}
    amr['Earthquake'] = {}
    amr['1976_Tangshan_earthquake'] = {}
    amr['e'] = {'wiki': [('1976_Tangshan_earthquake',)], 'name': [('n5',)]}
    amr['7'] = {}
    amr['d3'] = {'month': [('7',)], 'day': [('28',)]}
    amr['28'] = {}
    amr['m'] = {'ARG2': [('d3',)]}
    amr['d2'] = {'time-of': [('e',)], 'ARG1-of': [('m',)]}
    amr['p5'] = {'wiki': [('Mao_Zedong',)], 'name': [('n4',)]}
    amr['q'] = {'poss': [('p5',)]}
    amr['d'] = {'time': [('d2',)], 'location': [('q',)]}
    amr['Wang'] = {}
    amr['n'] = {'op1': [('Wang',)], 'op2': [('Dongxing',)]}
    amr['Dongxing'] = {}
    amr['Wang_Dongxing'] = {}
    amr['p2'] = {'wiki': [('Wang_Dongxing',)], 'name': [('n',)]}
    amr['n2'] = {'op1': [('Wang',)], 'op2': [('Hongwen',)]}
    amr['Hongwen'] = {}
    amr['Wang_Hongwen'] = {}
    amr['p3'] = {'wiki': [('Wang_Hongwen',)], 'name': [('n2',)]}
    amr['Zhang'] = {}
    amr['n3'] = {'op1': [('Zhang',)], 'op2': [('Yufeng',)]}
    amr['Yufeng'] = {}
    amr['Mao'] = {}
    amr['n4'] = {'op1': [('Mao',)], 'op2': [('Zedong',)]}
    amr['Zedong'] = {}
    amr['Mao_Zedong'] = {}
    amr['c'] = {}
    amr['s'] = {'mod': [('c',)]}
    amr['h'] = {'ARG1': [('p5',)], 'ARG2': [('s',)]}
    amr['-'] = {}
    amr['p4'] = {'wiki': [('-',)], 'name': [('n3',)], 'ARG0-of': [('h',)]}
    amr['a'] = {'op1': [('p2',)], 'op2': [('p3',)], 'op3': [('p4',)]}
    amr['p'] = {'prep-on': [('d',)], 'domain': [('a',)]}
    generated_amr, generated_sentence = train_pre_processing(amr, sentence)
    # expected
    # maybe I should replace it with PERSON1, PERSON2 or smth, I am losing a lot of info like this
    expected_sentence = 'On the day of the Tangshan Earthquake , i.e. July 28 th , those on duty at PERSON \'s ' \
                        'quarters ' \
                        'were PERSON , PERSON , and PERSON \'s confidential secretary , PERSON .'
    expected_amr: AMR = AMR()
    expected_amr.roots = ['p']
    expected_amr.reentrance_triples = [('q', 'poss', 'p5')]
    expected_amr.node_to_concepts = {'n5': 'name', 'e': 'earthquake', 'd3': 'date-entity', 'm': 'mean-01',
                                     'd2': 'day', 'q': 'quarter', 'd': 'duty', 'p2': 'PERSON',
                                     'p3': 'PERSON', 'p5': 'PERSON',
                                     'c': 'confidence', 's': 'secretary', 'h': 'have-org-role-91', 'p4': 'PERSON',
                                     'a': 'and', 'p': 'person'}
    # todo: maybe make sure the values in the alignment list are unique
    expected_amr.node_to_tokens = {'Tangshan': [('5', 'n5')], 'Earthquake': [('6', 'n5')], '7': [('9', 'd3')],
                                   '28': [('10', 'd3')], 'p5': ['17', '17'], 'd2': ['2'], 'q': ['19'],
                                   'p2': ['21'], 'p3': ['23'],
                                   'p4': ['31'], 's': ['29'],
                                   'h': ['27'], 'd': ['15'], 'a': ['25']}
    expected_amr.node_to_tokens = {'Tangshan': [('5', 'n5')], 'Earthquake': [('6', 'n5')], '7': [('9', 'd3')],
                                   '28': [('10', 'd3')], 'p5': ['17', '17'], 'd2': ['2'], 'q': ['19'],
                                   's': ['29'], 'h': ['27'], 'd': ['15'], 'a': ['25'], 'p2': ['21'],
                                   'p3': ['23'], 'p4': ['31']}
    expected_amr.relation_to_tokens = {'month': [('9', 'd3')], 'time-of': [('3', 'd2')], 'poss': [('18', 'q')],
                                       'location': [('16', 'd')], 'prep-on': [('0', 'p'), ('14', 'p')],
                                       'domain': [('8', 'p'), ('18', 'p'), ('20', 'p')]}
    expected_amr['Tangshan'] = {}
    expected_amr['n5'] = {'op1': [('Tangshan',)], 'op2': [('Earthquake',)]}
    expected_amr['Earthquake'] = {}
    expected_amr['1976_Tangshan_earthquake'] = {}
    expected_amr['e'] = {'wiki': [('1976_Tangshan_earthquake',)], 'name': [('n5',)]}
    expected_amr['7'] = {}
    expected_amr['d3'] = {'month': [('7',)], 'day': [('28',)]}
    expected_amr['28'] = {}
    expected_amr['m'] = {'ARG2': [('d3',)]}
    expected_amr['d2'] = {'time-of': [('e',)], 'ARG1-of': [('m',)]}
    expected_amr['p5'] = {}
    expected_amr['q'] = {'poss': [('p5',)]}
    expected_amr['d'] = {'time': [('d2',)], 'location': [('q',)]}
    expected_amr['p2'] = {}
    expected_amr['p3'] = {}
    expected_amr['c'] = {}
    expected_amr['s'] = {'mod': [('c',)]}
    expected_amr['h'] = {'ARG1': [('p5',)], 'ARG2': [('s',)]}
    expected_amr['p4'] = {'ARG0-of': [('h',)]}
    expected_amr['a'] = {'op1': [('p2',)], 'op2': [('p3',)], 'op3': [('p4',)]}
    expected_amr['p'] = {'prep-on': [('d',)], 'domain': [('a',)]}
    assert generated_sentence == expected_sentence
    assert_amr_graph_dictionaries(expected_amr, generated_amr)


# ::id DF-170-181103-888_4397.1 ::amr-annotator LDC-AMR-14 ::preferred
# ::tok It is Santorum that is the by far major nonRomney candidate and Newt would appear to be the spoiler .
# (a / and~e.11
#       :op1 (c / candidate~e.10
#             :ARG1-of (m / major-02~e.8
#                   :degree (b / by-far~e.6,7))
#             :mod (p3 / person :polarity - :wiki "Mitt_Romney"
#                   :name (n2 / name :op1 "Romney"))
#             :domain~e.1,4 (p2 / person :wiki "Rick_Santorum"
#                   :name (n / name :op1 "Santorum"~e.2)))
#       :op2 (a2 / appear-02~e.14
#             :ARG1 (s / spoil-01
#                   :ARG0 (p4 / person :wiki "Newt_Gingrich"
#                         :name (n3 / name :op1 "Newt"~e.12)))))
def test_train_pre_processing_ex_person_with_polarity():
    sentence = 'It is Santorum that is the by far major nonRomney candidate and Newt would appear to be the spoiler .'
    amr: AMR = AMR()
    amr.roots = ['a']
    amr.reentrance_triples = []
    amr.node_to_concepts = {'b': 'by-far', 'm': 'major-02', 'n2': 'name', 'p3': 'person', 'n': 'name',
                            'p2': 'person', 'c': 'candidate', 'n3': 'name', 'p4': 'person', 's': 'spoil-01',
                            'a2': 'appear-02', 'a': 'and'}
    amr.node_to_tokens = {'b': ['6', '7'], 'Santorum': [('2', 'n')], 'm': ['8'], 'Newt': [('12', 'n3')],
                          'c': ['10'], 'a2': ['14'], 'a': ['11']}
    amr.relation_to_tokens = {'domain': [('1', 'c'), ('4', 'c')]}
    amr['b'] = {}
    amr['m'] = {'degree': [('b',)]}
    amr['Romney'] = {}
    amr['n2'] = {'op1': [('Romney',)]}
    amr['-'] = {}
    amr['p3'] = {'polarity': [('-',)], 'wiki': [('Mitt_Romney',)], 'name': [('n2',)]}
    amr['Mitt_Romney'] = {}
    amr['Santorum'] = {}
    amr['n'] = {'op1': [('Santorum',)]}
    amr['Rick_Santorum'] = {}
    amr['p2'] = {'wiki': [('Rick_Santorum',)], 'name': [('n',)]}
    amr['c'] = {'ARG1-of': [('m',)], 'mod': [('p3',)], 'domain': [('p2',)]}
    amr['Newt'] = {}
    amr['n3'] = {'op1': [('Newt',)]}
    amr['Newt_Gingrich'] = {}
    amr['p4'] = {'wiki': [('Newt_Gingrich',)], 'name': [('n3',)]}
    amr['s'] = {'ARG0': [('p4',)]}
    amr['a2'] = {'ARG1': [('s',)]}
    amr['a'] = {'op1': [('c',)], 'op2': [('a2',)]}
    generated_amr, generated_sentence = train_pre_processing(amr, sentence)
    # expected
    expected_sentence = 'It is PERSON that is the by far major nonRomney candidate and PERSON would appear to be the spoiler .'
    expected_amr: AMR = AMR()
    expected_amr.roots = ['a']
    expected_amr.reentrance_triples = []
    expected_amr.node_to_concepts = {'b': 'by-far', 'm': 'major-02', 'n2': 'name', 'p3': 'person',
                                     'p2': 'PERSON', 'c': 'candidate', 'p4': 'PERSON', 's': 'spoil-01',
                                     'a2': 'appear-02', 'a': 'and'}
    expected_amr.node_to_tokens = {'b': ['6', '7'], 'p2': ['2'], 'm': ['8'], 'p4': ['12'],
                                   'c': ['10'], 'a2': ['14'], 'a': ['11']}
    expected_amr.relation_to_tokens = {'domain': [('1', 'c'), ('4', 'c')]}
    expected_amr['b'] = {}
    expected_amr['m'] = {'degree': [('b',)]}
    expected_amr['Romney'] = {}
    expected_amr['n2'] = {'op1': [('Romney',)]}
    expected_amr['-'] = {}
    expected_amr['p3'] = {'polarity': [('-',)], 'wiki': [('Mitt_Romney',)], 'name': [('n2',)]}
    expected_amr['Mitt_Romney'] = {}
    expected_amr['p2'] = {}
    expected_amr['c'] = {'ARG1-of': [('m',)], 'mod': [('p3',)], 'domain': [('p2',)]}
    expected_amr['p4'] = {}
    expected_amr['s'] = {'ARG0': [('p4',)]}
    expected_amr['a2'] = {'ARG1': [('s',)]}
    expected_amr['a'] = {'op1': [('c',)], 'op2': [('a2',)]}
    assert expected_sentence == generated_sentence
    assert_amr_graph_dictionaries(expected_amr, generated_amr)


if __name__ == "__main__":
    # test_train_pre_processing_ex_person()
    # test_train_pre_processing_ex_organization()
    # test_train_pre_processing_ex_2_person()
    # test_train_pre_processing_ex_person_reentrancy()
    # test_train_pre_processing_ex_3_person()
    # train_pre_processing_ex_persons_with_common_tokens()
    test_train_pre_processing_ex_person_with_polarity()
