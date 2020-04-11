from models.amr_data import CustomizedAMR
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
    custom_amr: CustomizedAMR = CustomizedAMR()
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
    generated_concepts = IdentifiedConcepts()
    generated_concepts.create_from_custom_amr('amr_id_1', custom_amr)
    expected_concepts = IdentifiedConcepts()
    expected_concepts.amr_id = 'amr_id_1'
    expected_concepts.ordered_concepts = [Concept('i', 'it'), Concept('r', 'recommend-01'), Concept('v', 'vigorous'),
                                          Concept('a', 'advocate-01')]
    assert_identified_concepts(expected_concepts, generated_concepts)


def test_create_from_custom_amr():
    test_create_from_custom_amr_example_1()


if __name__ == "__main__":
    test_create_from_custom_amr()
    print("Everything in concept_test passed")
