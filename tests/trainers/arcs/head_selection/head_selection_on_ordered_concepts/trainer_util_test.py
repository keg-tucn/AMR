from models.concept import IdentifiedConcepts, Concept
from models.node import Node
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.arcs_trainer_util import \
    generate_amr_node_for_parents_vector, calculate_smatch, generate_amr_node_for_vector_of_parents


# amr_str = """(r / recommend-01~e.1
#                 :ARG1 (a / advocate-01~e.4
#                     :ARG1 (i / it~e.0)
#                     :manner~e.2 (v / vigorous~e.3)))"""
# sentence = """It should be vigorously advocated ."""
def test_generate_amr_node_for_predicted_parents():
    identified_concepts: IdentifiedConcepts = IdentifiedConcepts()
    identified_concepts.ordered_concepts = [Concept('', 'ROOT'),  # 0
                                            Concept('i', 'it'),  # 1
                                            Concept('r', 'recommend-01'),  # 2
                                            Concept('v', 'vigorous'),  # 3
                                            Concept('a', 'advocate-01')  # 4
                                            ]
    predicted_parents = [-1, 4, 0, 4, 2]
    relations_dict = {
        ('recommend-01', 'advocate-01'): 'ARG1',
        ('advocate-01', 'it'): 'ARG1',
        ('advocate-01', 'vigorous'): 'manner'
    }
    amr: Node = generate_amr_node_for_parents_vector(identified_concepts,
                                                     predicted_parents,
                                                     relations_dict)
    generated_amr_str = amr.amr_print()
    gold_amr_str = """(r / recommend-01~e.1
                    :ARG1 (a / advocate-01~e.4
                        :ARG1 (i / it~e.0)
                        :manner~e.2 (v / vigorous~e.3)))"""
    smatch = calculate_smatch(generated_amr_str, gold_amr_str)
    assert smatch == 1


# amr_str = """(r / recommend-01~e.1
#                 :ARG1 (a / advocate-01~e.4
#                     :ARG1 (i / it~e.0)
#                     :manner~e.2 (v / vigorous~e.3)))"""
# sentence = """It should be vigorously advocated ."""
def test_generate_amr_node_for_vector_of_parents():
    identified_concepts: IdentifiedConcepts = IdentifiedConcepts()
    identified_concepts.ordered_concepts = [Concept('', 'ROOT'),  # 0
                                            Concept('i', 'it'),  # 1
                                            Concept('r', 'recommend-01'),  # 2
                                            Concept('v', 'vigorous'),  # 3
                                            Concept('a', 'advocate-01')  # 4
                                            ]
    predicted_vector_of_parents = [[-1], [4], [0], [4], [2]]
    relations_dict = {
        ('recommend-01', 'advocate-01'): 'ARG1',
        ('advocate-01', 'it'): 'ARG1',
        ('advocate-01', 'vigorous'): 'manner'
    }
    amr: Node = generate_amr_node_for_vector_of_parents(identified_concepts,
                                                        predicted_vector_of_parents,
                                                        relations_dict)
    generated_amr_str = amr.amr_print_with_reentrancy()
    gold_amr_str = """(r / recommend-01~e.1
                    :ARG1 (a / advocate-01~e.4
                        :ARG1 (i / it~e.0)
                        :manner~e.2 (v / vigorous~e.3)))"""
    smatch = calculate_smatch(generated_amr_str, gold_amr_str)
    assert smatch == 1


if __name__ == "__main__":
    test_generate_amr_node_for_predicted_parents()
    test_generate_amr_node_for_vector_of_parents()
    print("Everything in dummy_util passed")
