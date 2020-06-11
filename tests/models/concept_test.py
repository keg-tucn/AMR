from models.amr_graph import AMR
from models.concept import IdentifiedConcepts, Concept


def assert_identified_concepts(expected_concepts: IdentifiedConcepts, generated_concepts: IdentifiedConcepts):
    assert expected_concepts.amr_id == generated_concepts.amr_id, \
        'amr_id' + str(generated_concepts.amr_id) + 'should be' + str(expected_concepts.amr_id)
    assert expected_concepts.ordered_concepts == generated_concepts.ordered_concepts, \
        'ordered_concepts' + str(generated_concepts.ordered_concepts) + 'should be' + str(
            expected_concepts.ordered_concepts)


# amr_str = """(r / recommend-01~e.1
#                 :ARG1 (a / advocate-01~e.4
#                     :ARG1 (i / it~e.0)
#                     :manner~e.2 (v / vigorous~e.3)))"""
# sentence = """It should be vigorously advocated ."""
def test_create_from_custom_amr_example_1():
    amr: AMR = AMR()
    amr.node_to_concepts = {'i': 'it', 'v': 'vigorous', 'a': 'advocate-01', 'r': 'recommend-01'}
    amr.node_to_tokens = {'i': ['0'], 'v': ['3'], 'a': ['4'], 'r': ['1']}
    amr.relation_to_tokens = {'manner': [('2', 'a')]}
    amr['i'] = {}
    amr['v'] = {}
    amr['a'] = {'ARG1': [('i',)], 'manner': [('v',)]}
    amr['r'] = {'ARG1': [('a',)]}
    generated_concepts = IdentifiedConcepts()
    generated_concepts.create_from_amr('amr_id_1', amr)
    expected_concepts = IdentifiedConcepts()
    expected_concepts.amr_id = 'amr_id_1'
    expected_concepts.ordered_concepts = [Concept('i', 'it'), Concept('r', 'recommend-01'), Concept('v', 'vigorous'),
                                          Concept('a', 'advocate-01')]
    assert_identified_concepts(expected_concepts, generated_concepts)


def test_create_from_custom_amr():
    test_create_from_custom_amr_example_1()


def test_strip_concept_sense():
    concept_name = 'recommend-01'
    stripped_concept = Concept.strip_concept_sense(concept_name)
    expected_concept = 'recommend'
    assert stripped_concept == expected_concept

    concept_name = 'go'
    stripped_concept = Concept.strip_concept_sense(concept_name)
    expected_concept = 'go'
    assert stripped_concept == expected_concept

    concept_name = '-'
    stripped_concept = Concept.strip_concept_sense(concept_name)
    expected_concept = '-'
    assert stripped_concept == expected_concept


if __name__ == "__main__":
    test_create_from_custom_amr()
    test_strip_concept_sense()
    print("Everything in concept_test passed")
