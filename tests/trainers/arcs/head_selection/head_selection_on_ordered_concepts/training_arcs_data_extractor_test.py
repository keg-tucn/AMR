from typing import List
from models.amr_data import CustomizedAMR
from models.amr_graph import AMR
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


#
# amr_str = """(r / recommend-01~e.1
#                 :ARG1 (a / advocate-01~e.4
#                     :ARG1 (i / it~e.0)
#                     :manner~e.2 (v / vigorous~e.3)))"""
# sentence = """It should be vigorously advocated ."""
def test_generate_parent_vector():
    amr: AMR = AMR()
    amr.roots = ['r']
    amr.reentrance_triples = []
    amr.node_to_concepts = {'i': 'it', 'v': 'vigorous', 'a': 'advocate-01', 'r': 'recommend-01'}
    amr.node_to_tokens = {'i': ['0'], 'v': ['3'], 'a': ['4'], 'r': ['1']}
    amr.relation_to_tokens = {'manner': [('2', 'a')]}
    amr['i'] = {}
    amr['v'] = {}
    amr['a'] = {'ARG1': [('i',)], 'manner': [('v',)]}
    amr['r'] = {'ARG1': [('a',)]}
    identified_concepts = IdentifiedConcepts()
    identified_concepts.ordered_concepts = [Concept('', 'ROOT'),
                                            Concept('i', 'it'),
                                            Concept('r', 'recommend-01'),
                                            Concept('v', 'vigorous'),
                                            Concept('a', 'advocate-01')]
    generated_parent_vector = generate_parent_vectors(amr, identified_concepts, 1)
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
    generated_entry: ArcsTrainingEntry = generate_dataset_entry('amr_id', amr_str, sentence, 0, 1, False, False)
    expected_identified_concepts = IdentifiedConcepts()
    expected_identified_concepts.amr_id = 'amr_id'
    expected_identified_concepts.ordered_concepts = [Concept('', 'ROOT'),
                                                     Concept('i', 'it'),
                                                     Concept('r', 'recommend-01'),
                                                     Concept('v', 'vigorous'),
                                                     Concept('a', 'advocate-01')]
    expected_parent_vectors = [(-1, 4, 0, 4, 2)]
    assert_identified_concepts(expected_identified_concepts, generated_entry.identified_concepts)
    assert_parent_vectors(expected_parent_vectors, generated_entry.parent_vectors)


if __name__ == "__main__":
    test_generate_parent_vector()
    test_generate_dataset_entry()
    print("Everything in dataset_generator_test passed")
