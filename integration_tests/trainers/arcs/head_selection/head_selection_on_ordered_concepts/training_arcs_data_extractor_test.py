from typing import List
from models.amr_graph import AMR
from models.concept import IdentifiedConcepts, Concept
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.parents_vector_extractor import \
    generate_parent_vectors


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
def test_generate_parent_vector_example_1():
    amr_str = """(r / recommend-01~e.1
                    :ARG1 (a / advocate-01~e.4
                        :ARG1 (i / it~e.0)
                        :manner~e.2 (v / vigorous~e.3)))"""
    amr: AMR = AMR.parse_string(amr_str)
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
def test_generate_parent_vector_example_2():
    amr_str = """(r / recommend-01~e.1
                    :ARG1 (a / advocate-01~e.4
                        :ARG1 (i / it~e.0)
                        :manner~e.2 (v / vigorous~e.3)))"""
    amr: AMR = AMR.parse_string(amr_str)
    identified_concepts = IdentifiedConcepts()
    identified_concepts.ordered_concepts = [Concept('', 'ROOT'),
                                            Concept('i', 'it'),
                                            Concept('r', 'recommend-01'),
                                            Concept('v', 'vigorous'),
                                            Concept('a', 'advocate-01')]
    generated_parent_vector = generate_parent_vectors(amr, identified_concepts)
    expected_parent_vector = [[-1, 4, 0, 4, 2]]
    assert_parent_vectors(expected_parent_vector, generated_parent_vector)


# AMR with id DF-199-194209-587_2515.12
def test_generate_parent_vector_example_2():
    amr_str = """(m / man~e.2 
      :ARG1-of (m2 / marry-01~e.1) 
      :ARG0-of (l / love-01~e.9 
            :ARG1~e.10 (y / you~e.11) 
            :ARG1-of (r / real-04~e.6) 
            :condition-of~e.4 (a3 / and~e.16 
                  :op1 (g / go-06~e.14 
                        :ARG2 (a / ahead~e.15) 
                        :mod (j / just~e.13)) 
                  :op2 (o2 / or~e.22 
                        :op1 (f / file-01~e.17 
                              :ARG4~e.18 (d / divorce-01~e.19) 
                              :time (n / now~e.20)) 
                        :op2 (m3 / move-01~e.25 
                              :ARG2 (o / out-06~e.26 
                                    :ARG2~e.27 (h / house~e.29 
                                          :poss~e.28 m~e.28)) 
                              :time n~e.30 
                              :mod (a2 / at-least~e.23,24))))))"""
    amr: AMR = AMR.parse_string(amr_str)
    identified_concepts = IdentifiedConcepts()
    identified_concepts.ordered_concepts = [Concept('', 'ROOT'),  # 0
                                            Concept('m2', 'marry-01'),  # 1
                                            Concept('m', 'man'),  # 2
                                            Concept('r', 'real-04'),  # 3
                                            Concept('l', 'love-01'),  # 4
                                            Concept('y', 'you'),  # 5
                                            Concept('j', 'just'),  # 6
                                            Concept('g', 'go-06'),  # 7
                                            Concept('a', 'ahead'),  # 8
                                            Concept('a3', 'and'),  # 9
                                            Concept('f', 'file-01'),  # 10
                                            Concept('d', 'divorce-01'),  # 11
                                            Concept('n', 'now'),  # 12
                                            Concept('o2', 'or'),  # 13
                                            Concept('a2', 'at-least'),  # 14
                                            Concept('m3', 'move-01'),  # 15
                                            Concept('o', 'out-06'),  # 16
                                            Concept('h', 'house')  # 17
                                            ]
    generated_parent_vector = generate_parent_vectors(amr, identified_concepts, 2)
    expected_parent_vector = [(-1, 2, 0, 4, 2, 4, 7, 9, 7, 4, 13, 10, 10, 9, 15, 13, 15, 16),
                              (-1, 2, 0, 4, 2, 4, 7, 9, 7, 4, 13, 10, 15, 9, 15, 13, 15, 16)]
    assert_parent_vectors(expected_parent_vector, generated_parent_vector)

# TODO: test for bolt12_07_4800.2

if __name__ == "__main__":
    test_generate_parent_vector_example_1()
    test_generate_parent_vector_example_2()
    print("Everything in dataset_generator_test passed")
