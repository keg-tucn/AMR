"""
Methods that set up the AMR from smatch score processing.
Please see some examples in the main.
"""


import smatch
import smatch_amr as amr
import copy


def smatch_f_score(amr1, amr2,
                   doinstance=True, doattribute=True, dorelation=True):
    """
    The "best match" number is the number of matching nodes. Larger for larger matching AMRs.
    :param amr1: expected in smatch_amr format
    :param amr2: expected in smatch_amr format
    """

    clean_all_node_names(amr1)
    clean_all_node_names(amr2)

    prefix1 = "a"
    prefix2 = "b"
    # Rename node to "a1", "a2", .etc
    amr1.rename_node(prefix1)
    # Renaming node to "b1", "b2", .etc
    amr2.rename_node(prefix2)

    (instance1, attributes1, relation1) = amr1.get_triples()
    (instance2, attributes2, relation2) = amr2.get_triples()

    # without this line we sometimes get smatch scores greater than 1, probably an incorrect caching of some sort
    smatch.match_triple_dict.clear()

    (best_mapping, best_match_num) = smatch.get_best_match(
        instance1, attributes1, relation1,
        instance2, attributes2, relation2,
        prefix1, prefix2,
        doinstance=doinstance, doattribute=doattribute, dorelation=dorelation)

    test_triple_num = 0
    gold_triple_num = 0
    if doinstance:
        test_triple_num += len(instance1)
        gold_triple_num += len(instance2)
    if doattribute:
        test_triple_num += len(attributes1)
        gold_triple_num += len(attributes2)
    if dorelation:
        test_triple_num += len(relation1)
        gold_triple_num += len(relation2)

    (precision, recall, best_f_score) = smatch.compute_f(
        best_match_num, test_triple_num, gold_triple_num)

    return best_f_score


def clean_node_value(nv):
    return nv.strip(' \t\n\r').split('~', 1)[0]


def clean_all_node_names(amr):
    updated_node_values = [clean_node_value(nv) for nv in amr.node_values]
    amr.node_values = updated_node_values
    updated_attributes = [[[clean_node_value(tok) for tok in toks] for toks in nattr] for nattr in amr.attributes]
    amr.attributes = updated_attributes
    updated_relations = [[[clean_node_value(tok) for tok in toks] for toks in nrel] for nrel in amr.relations]
    amr.relations = updated_relations

if __name__ == "__main__":
    smatch.veryVerbose=False

    str1 = """
    (n / need-01~e.2 
      :ARG0 (y / you~e.0) 
      :manner~e.1 (r / real~e.1))
    """
    amr1 = amr.AMR.parse_AMR_line(str1)

    print smatch_f_score(amr1, copy.deepcopy(amr1))


    str2="""
    ( d3 / need-01 
	:manner  ( d2 / real )
	:ARG0  ( d1 / you )
    """
    amr2 = amr.AMR.parse_AMR_line(str2)

    print smatch_f_score(amr1, amr2)
