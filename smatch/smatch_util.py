"""
Methods that set up the AMR from smatch score processing.
Please see some examples in the main.
"""


import smatch
import smatch_amr as amr
import numpy as np
import copy


def smatch_best_match_numbers(amr1, amr2,
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

    return best_match_num, test_triple_num, gold_triple_num


def smatch_f_score(amr1, amr2,
                   doinstance=True, doattribute=True, dorelation=True):

    best_match_num, test_triple_num, gold_triple_num = smatch_best_match_numbers(amr1, amr2,
                   doinstance, doattribute, dorelation)
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


class SmatchAccumulator:

    def __init__(self):
        self.n = 0
        self.smatch_scores = []
        self.total_match_num = 0
        self.total_amr1_num = 0
        self.total_amr2_num = 0
        self.smatch_sum = 0
        self.inv_smatch_sum = 0
        self.last_f_score = 0;

    def compute_and_add(self, amr1, amr2):
        self.n += 1

        best_match_num, test_triple_num, gold_triple_num = smatch_best_match_numbers(amr1, amr2)
        self.total_match_num += best_match_num
        self.total_amr1_num += test_triple_num
        self.total_amr2_num += gold_triple_num

        (precision, recall, self.last_f_score) = smatch.compute_f(
            best_match_num, test_triple_num, gold_triple_num)

        self.smatch_scores.append(self.last_f_score)
        self.smatch_sum += self.last_f_score
        self.inv_smatch_sum += 1 / self.last_f_score

        return self.last_f_score

    def smatch_mean(self):
        return self.n / self.inv_smatch_sum

    def smatch_per_node_mean(self):
        _, _, best_f_score = smatch.compute_f(
            self.total_match_num, self.total_amr1_num, self.total_amr2_num)
        return best_f_score

    def print_all(self):
        if self.n == 0:
          print ("No results")
        else:
            print("Min: %f" % np.min(self.smatch_scores))
            print("Max: %f" % np.max(self.smatch_scores))
            print ("Arithm. mean %s" % (self.smatch_sum / self.n))
            print ("Harm. mean %s" % (self.n / self.inv_smatch_sum))
            print ("Global match f-score %s" % self.smatch_per_node_mean())


if __name__ == "__main__":
    smatch.veryVerbose=False

    str1 = """
    ( c / chat-01~e.6 
	:ARG0  ( i / i~e.0 )
	:ARG1  ( b / behave-01 
		:ARG0  ( h / he~e.8,10 ))
	:ARG2~e.7  h)
	:mod  ( a / also~e.2 ))
    """
    amr1 = amr.AMR.parse_AMR_line(str1)
    #clean_all_node_names(amr1)
    print amr1.pretty_print()

    #print smatch_f_score(amr1, copy.deepcopy(amr1))


    str2="""
    (c / contrast-01~e.0 
      :ARG2 (n / need-01~e.15 
            :mod (a2 / also~e.14) 
            :ARG1 (r / restart-01~e.10 
                  :ARG1 (d / dialogue-01~e.12)) 
            :prep-as~e.2 (m / method~e.5 
                  :mod (a3 / auxiliary~e.4) 
                  :instrument-of~e.6 (m2 / measure-02~e.8 
                        :ARG0 (m3 / military~e.7)))))
    """
    amr2 = amr.AMR.parse_AMR_line(str2)
    clean_all_node_names(amr2)
    print amr2.pretty_print()

    print smatch_f_score(amr1, amr2)
