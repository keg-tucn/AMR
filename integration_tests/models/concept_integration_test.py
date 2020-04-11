# amr_str = """(r / recommend-01~e.1
#                 :ARG1 (a / advocate-01~e.4
#                     :ARG1 (i / it~e.0)
#                     :manner~e.2 (v / vigorous~e.3)))"""
# sentence = """It should be vigorously advocated ."""
from models.amr_data import CustomizedAMR
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
    amr_str = """(r / recommend-01~e.1
                    :ARG1 (a / advocate-01~e.4
                        :ARG1 (i / it~e.0)
                        :manner~e.2 (v / vigorous~e.3)))"""
    amr = AMR.parse_string(amr_str)
    custom_amr = CustomizedAMR()
    custom_amr.create_custom_AMR(amr)
    generated_concepts = IdentifiedConcepts()
    generated_concepts.create_from_custom_amr('amr_id_1', custom_amr)
    expected_concepts = IdentifiedConcepts()
    expected_concepts.amr_id = 'amr_id_1'
    expected_concepts.ordered_concepts = [Concept('i', 'it'), Concept('r', 'recommend-01'), Concept('v', 'vigorous'),
                                          Concept('a', 'advocate-01')]
    assert_identified_concepts(expected_concepts, generated_concepts)


# ::id bolt12_10510_9791.9 ::amr-annotator SDL-AMR-09 ::preferred
# ::tok And by not issuing any promises , he could actually avoid being censured .
# ::alignments 0-1 2-1.1.3.1 2-1.1.3.1.r 4-1.1.3.3 5-1.1.3 7-1.1.1.1 8-1.1 9-1.1.2 10-1.1.1 12-1.1.1.2
# (a / and~e.0
#       :op2 (p / possible-01~e.8
#             :ARG1 (a3 / avoid-01~e.10
#                   :ARG0 (h / he~e.7)
#                   :ARG1 (c / censure-01~e.12
#                         :ARG1 h))
#             :ARG1-of (a2 / actual-02~e.9)
#             :manner (p2 / promise-01~e.5 :polarity~e.2 -~e.2
#                   :ARG0 h
#                   :mod (a4 / any~e.4))))
def test_create_from_custom_amr_example_2():
    amr_str = """(a / and~e.0 
      :op2 (p / possible-01~e.8 
            :ARG1 (a3 / avoid-01~e.10 
                  :ARG0 (h / he~e.7) 
                  :ARG1 (c / censure-01~e.12 
                        :ARG1 h)) 
            :ARG1-of (a2 / actual-02~e.9) 
            :manner (p2 / promise-01~e.5 :polarity~e.2 -~e.2 
                  :ARG0 h 
                  :mod (a4 / any~e.4))))"""
    amr = AMR.parse_string(amr_str)
    custom_amr = CustomizedAMR()
    custom_amr.create_custom_AMR(amr)
    generated_concepts = IdentifiedConcepts()
    generated_concepts.create_from_custom_amr('amr_id_2', custom_amr)
    expected_concepts = IdentifiedConcepts()
    expected_concepts.amr_id = 'amr_id_2'
    expected_concepts.ordered_concepts = [Concept('a', 'and'), Concept('-', '-'), Concept('a4', 'any'),
                                          Concept('p2', 'promise-01'), Concept('h', 'he'), Concept('p', 'possible-01'),
                                          Concept('a2', 'actual-02'), Concept('a3', 'avoid-01'), Concept('c', 'censure-01')]
    assert_identified_concepts(expected_concepts, generated_concepts)


def test_create_from_custom_amr():
    test_create_from_custom_amr_example_1()
    test_create_from_custom_amr_example_2()


if __name__ == "__main__":
    test_create_from_custom_amr()
    print("Everything in concept_test passed")
