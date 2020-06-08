from typing import List
from models.amr_data import CustomizedAMR
from models.concept import IdentifiedConcepts, Concept
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.parents_vector_extractor import \
    generate_parent_vectors
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.training_arcs_data_extractor import \
    generate_dataset_entry, ArcsTrainingEntry


# TODO: more general method for assertions (or use test framework)
def assert_parent_vectors(expected_vector: List, generated_vector: List):
    assert expected_vector == generated_vector, \
        'parent_vector ' + str(generated_vector) + ' should be ' + str(expected_vector)


def assert_identified_concepts(expected_concepts: IdentifiedConcepts, generated_concepts: IdentifiedConcepts):
    assert expected_concepts == generated_concepts, \
        'concepts ' + str(generated_concepts) + ' should be ' + str(expected_concepts)


# amr_str = """(r / recommend-01~e.1
#                 :ARG1 (a / advocate-01~e.4
#                     :ARG1 (i / it~e.0)
#                     :manner~e.2 (v / vigorous~e.3)))"""
# sentence = """It should be vigorously advocated ."""
def test_generate_parent_vector():
    custom_amr = CustomizedAMR()
    custom_amr.tokens_to_concepts_dict = {0: ('i', 'it'),
                                          1: ('r', 'recommend-01'),
                                          3: ('v', 'vigorous'),
                                          4: ('a', 'advocate-01')}
    custom_amr.tokens_to_concept_list_dict = {0: [('i', 'it')],
                                              1: [('r', 'recommend-01')],
                                              3: [('v', 'vigorous')],
                                              4: [('a', 'advocate-01')]}
    # (child,parent) : (relation, children of child, token aligned to child)
    custom_amr.relations_dict = {('i', 'a'): ('ARG1', [], ['0']),
                                 ('v', 'a'): ('manner', [], ['3']),
                                 ('r', ''): ('', ['a'], ['1']),
                                 ('a', 'r'): ('ARG1', ['i', 'v'], ['4'])}
    custom_amr.parent_dict = {'i': 'a', 'v': 'a', 'a': 'r', 'r': ''}
    identified_concepts = IdentifiedConcepts()
    identified_concepts.ordered_concepts = [Concept('', 'ROOT'),
                                            Concept('i', 'it'),
                                            Concept('r', 'recommend-01'),
                                            Concept('v', 'vigorous'),
                                            Concept('a', 'advocate-01')]
    generated_parent_vector = generate_parent_vectors(custom_amr, identified_concepts)
    expected_parent_vector = [(-1, 4, 0, 4, 2)]
    assert_parent_vectors(expected_parent_vector, generated_parent_vector)


# amr_str = """(r / recommend-01~e.1
#                 :ARG1 (a / advocate-01~e.4
#                     :ARG1 (i / it~e.0)
#                     :manner~e.2 (v / vigorous~e.3)))"""
# sentence = """It should be vigorously advocated ."""
def test_generate_dataset_entry():
    amr_str = """(r / recommend-01~e.1
                    :ARG1 (a / advocate-01~e.4
                        :ARG1 (i / it~e.0)
                        :manner~e.2 (v / vigorous~e.3)))"""
    sentence = """It should be vigorously advocated ."""
    generated_entry: ArcsTrainingEntry = generate_dataset_entry('amr_id', amr_str,sentence,0)
    expected_identified_concepts = IdentifiedConcepts()
    expected_identified_concepts.amr_id = 'amr_id'
    expected_identified_concepts.ordered_concepts = [Concept('', 'ROOT'),
                                                     Concept('i', 'it'),
                                                     Concept('r', 'recommend-01'),
                                                     Concept('v', 'vigorous'),
                                                     Concept('a', 'advocate-01')]
    expected_parent_vector = [-1, 4, 0, 4, 2]
    assert_identified_concepts(expected_identified_concepts, generated_entry.identified_concepts)
    assert_parent_vectors(expected_parent_vector, generated_entry.parent_vector)


if __name__ == "__main__":
    test_generate_parent_vector()
    test_generate_dataset_entry()
    print("Everything in dataset_generator_test passed")