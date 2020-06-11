from models.amr_graph import AMR
from models.concept import IdentifiedConcepts
from models.node import Node
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.parents_vector_extractor import \
    generate_parent_list_vector

from trainers.arcs.head_selection.head_selection_on_ordered_concepts.trainer_util import \
    generate_amr_node_for_vector_of_parents, calculate_smatch
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.training_arcs_data_extractor import \
    add_false_root


# ::id DF-199-194190-649_6415.18 ::amr-annotator SDL-AMR-09 ::preferred
# ::tok I suppose you could add probation ( but that is just a replacement for jail time ) .
# (s / suppose-01~e.1
#       :ARG0 (i / i~e.0)
#       :ARG1 (p / possible-01~e.3
#             :ARG1 (a / add-02~e.4
#                   :ARG0 (y / you~e.2)
#                   :ARG1 (p2 / probation~e.5
#                         :ARG1-of (c / contrast-01~e.7
#                               :ARG2 (r / replace-01~e.12
#                                     :ARG1 p2
#                                     :ARG2~e.13 (t / time~e.15
#                                           :mod (j / jail~e.14))
#                                     :mod (j2 / just~e.10)))))))
def test_generate_amr_node_for_vector_of_parents_example_1():
    amr_str = """(s / suppose-01~e.1 
                      :ARG0 (i / i~e.0) 
                      :ARG1 (p / possible-01~e.3 
                            :ARG1 (a / add-02~e.4 
                                  :ARG0 (y / you~e.2) 
                                  :ARG1 (p2 / probation~e.5 
                                        :ARG1-of (c / contrast-01~e.7 
                                              :ARG2 (r / replace-01~e.12 
                                                    :ARG1 p2 
                                                    :ARG2~e.13 (t / time~e.15 
                                                          :mod (j / jail~e.14)) 
                                                    :mod (j2 / just~e.10)))))))"""
    amr_str1 = """(d1 / suppose-01~e.1 
                      :ARG0 (i / i~e.0) 
                      :ARG1 (p / possible-01~e.3 
                            :ARG1 (a / add-02~e.4 
                                  :ARG0 (y / you~e.2) 
                                  :ARG1 (p2 / probation~e.5 
                                        :ARG1-of (c / contrast-01~e.7 
                                              :ARG2 (r / replace-01~e.12 
                                                    :ARG1 p2 
                                                    :mod (j2 / just~e.10)
                                                    :ARG2~e.13 (t / time~e.15 
                                                          :mod (j / jail~e.14)) 
                                                    ))))))"""
    amr: AMR = AMR.parse_string(amr_str)
    identified_concepts = IdentifiedConcepts()
    identified_concepts.create_from_amr('amr_1', amr)
    add_false_root(identified_concepts)
    vector_of_parents = generate_parent_list_vector(amr, identified_concepts)
    # transf parent vectors to vector of parents
    # i s y p a p2 c j2 r j  t
    # 1 2 3 4 5 6  7 8  9 10 11
    relations_dict = {('suppose-01', 'i'): 'ARG0',
                      ('suppose-01', 'possible-01'): 'ARG1',
                      ('possible-01', 'add-02'): 'ARG1',
                      ('add-02', 'you'): 'ARG0',
                      ('add-02', 'probation'): 'ARG1',
                      ('probation', 'contrast-01'): 'ARG1-of',
                      ('contrast-01', 'replace-01'): 'ARG2',
                      ('replace-01', 'probation'): 'ARG1',
                      ('replace-01', 'time'): 'ARG2',
                      ('replace-01', 'just'): 'mod',
                      ('time', 'jail'): 'mod'}
    amr_node: Node = generate_amr_node_for_vector_of_parents(identified_concepts, vector_of_parents, relations_dict)
    generated_amr_str = amr_node.amr_print_with_reentrancy()
    smatch = calculate_smatch(generated_amr_str, amr_str)
    assert smatch == 1


if __name__ == "__main__":
    test_generate_amr_node_for_vector_of_parents_example_1()
    print("Everything in trainer_util_integration_test passed")
