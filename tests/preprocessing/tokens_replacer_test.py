from models.amr_graph import AMR
from models.node import Node
from preprocessing.TokensReplacer import replace_named_entities


# ::id PROXY_APW_ENG_20080514_1125.7 ::amr-annotator SDL-AMR-09 ::preferred
# ::tok The center will bolster NATO 's defenses against cyber attacks .
# (b / bolster-01~e.3
#       :ARG0 (c / center~e.1)
#       :ARG1 (d / defend-01~e.6
#             :ARG0~e.5 (m / military :wiki "NATO"
#                   :name (n / name :op1 "NATO"~e.4))
#             :ARG1 m
#             :ARG3 (a / attack-01~e.9
#                   :medium (c2 / cyber~e.8))))
def test_replace_named_entities():
    sentence = "The center will bolster NATO 's defenses against cyber attacks ."
    amr_str = """(b / bolster-01~e.3 
      :ARG0 (c / center~e.1) 
      :ARG1 (d / defend-01~e.6 
            :ARG0~e.5 (m / military :wiki "NATO" 
                  :name (n / name :op1 "NATO"~e.4)) 
            :ARG1 m 
            :ARG3 (a / attack-01~e.9 
                  :medium (c2 / cyber~e.8))))"""
    amr: AMR = AMR()
    amr.node_to_concepts = {'c': 'center', 'n': 'name', 'm': 'military',
                            'c2': 'cyber', 'a': 'attack-01', 'd': 'defend-01', 'b': 'bolster-01'}
    amr.node_to_tokens = {'NATO': [('4', 'n')], 'c2': ['8'], 'a': ['9'], 'c': ['1'], 'd': ['6'], 'b': ['3']}
    amr.relation_to_tokens = {'ARG0': [('5', 'd')]}
    amr['c'] = {}
    amr["NATO"] = {}
    amr['n'] = {'op1': ["NATO"]}
    amr['m'] = {'wiki': ["NATO"], 'name': ['n']}
    amr['c2'] = {}
    amr['a'] = {'medium': ['c2']}
    amr['d'] = {'ARG0': ['m'], 'ARG1': ['m'], 'ARG3': ['a']}
    amr['b'] = {'ARG0': ['c'], 'ARG1': ['d']}
    amr.reentrance_triples = [('d', 'ARG1', 'm')]
    amr.roots = ['b']
    generated_amr, generated_sentence, generated_metadata = replace_named_entities(amr, sentence)
    generated_subgraph = generated_metadata[0][5]
    expected_sentence = "The center will bolster military 's defenses against cyber attacks ."
    expected_amr: AMR = AMR()
    expected_amr.node_to_concepts = {'c': 'center', 'm': 'military',
                                     'c2': 'cyber', 'a': 'attack-01', 'd': 'defend-01', 'b': 'bolster-01'}
    expected_amr.node_to_tokens = {'c2': ['8'], 'a': ['9'], 'c': ['1'], 'd': ['6'], 'b': ['3']}
    expected_amr.relation_to_tokens = {'ARG0': [('5', 'd')]}
    expected_amr['c'] = {}
    expected_amr['m'] = {}
    expected_amr['c2'] = {}
    expected_amr['a'] = {'medium': ['c2']}
    expected_amr['d'] = {'ARG0': ['m'], 'ARG1': ['m'], 'ARG3': ['a']}
    expected_amr['b'] = {'ARG0': ['c'], 'ARG1': ['d']}
    expected_amr.reentrance_triples = [('d', 'ARG1', 'm')]
    expected_amr.roots = ['b']
    # metadata
    expected_subgraph: Node = Node('military')
    n = Node('name')
    op1_literal = Node(None, '\"NATO\"')
    wiki_literal = Node(None, '\"NATO\"')
    n.add_child(op1_literal,'op1')
    expected_subgraph.add_child(n,'name')
    expected_subgraph.add_child(wiki_literal,'wiki')
    assert generated_sentence == expected_sentence
    assert generated_amr == expected_amr
    assert generated_subgraph.amr_print() == expected_subgraph.amr_print()
    assert generated_metadata[0][0] == 'm'
    assert generated_metadata[0][1] == 'n'
    assert generated_metadata[0][2] == ['NATO']
    assert generated_metadata[0][3] == 4
    assert generated_metadata[0][4] == 4


if __name__ == '__main__':
    test_replace_named_entities()
