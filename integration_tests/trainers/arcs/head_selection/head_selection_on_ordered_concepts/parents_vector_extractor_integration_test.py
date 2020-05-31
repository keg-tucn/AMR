from models.amr_graph import AMR
from models.concept import IdentifiedConcepts
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.parents_vector_extractor import \
    generate_parent_list_vector
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.training_arcs_data_extractor import add_false_root


# amr_str = """(r / recommend-01~e.1
#                 :ARG1 (a / advocate-01~e.4
#                     :ARG1 (i / it~e.0)
#                     :manner~e.2 (v / vigorous~e.3)))"""
# sentence = """It should be vigorously advocated ."""
def test_generate_parent_list_vector_ex_1():
    amr_str = """(r / recommend-01~e.1
                    :ARG1 (a / advocate-01~e.4
                        :ARG1 (i / it~e.0)
                        :manner~e.2 (v / vigorous~e.3)))"""
    amr = AMR.parse_string(amr_str)
    identified_concepts = IdentifiedConcepts()
    identified_concepts.create_from_amr('amr_id_1', amr)
    add_false_root(identified_concepts)
    generated_parent_list_vector = generate_parent_list_vector(amr, identified_concepts)
    # i r v a
    # 1 2 3 4
    expected_parent_list_vector = [[-1], [4], [0], [4], [2]]
    assertion_message = str(generated_parent_list_vector) + ' should be' + str(expected_parent_list_vector)
    assert generated_parent_list_vector == expected_parent_list_vector, assertion_message


# ::id bolt-eng-DF-170-181103-8889109_0043.4 ::amr-annotator UCO-AMR-05 ::preferred
# ::tok 2012 may be a year of unexpected recovery .
# amr_str = """(y2 / year~e.4
#                   :time-of~e.5 (r / recover-01~e.7
#                         :ARG1-of (e / expect-01 :polarity -~e.6))
#                   :ARG1-of (p / possible-01~e.1)
#                   :domain~e.2 (d / date-entity :year~e.4 2012~e.0))"""
def test_generate_parent_list_vector_with_polarity():
    amr_str = """(y2 / year~e.4
                      :time-of~e.5 (r / recover-01~e.7
                            :ARG1-of (e / expect-01~e.6 :polarity -~e.6))
                      :ARG1-of (p / possible-01~e.1)
                      :domain~e.2 (d / date-entity~e.4 :year~e.4 2012~e.0))"""
    amr: AMR = AMR.parse_string(amr_str)
    identified_concepts = IdentifiedConcepts()
    identified_concepts.create_from_amr('amr_polarity', amr)
    add_false_root(identified_concepts)
    generated_parent_list_vector = generate_parent_list_vector(amr, identified_concepts)
    # 2012 p d y2 - e r
    # 1    2 3 4  5 6 7
    expected_parent_list_vector = [[-1], [3], [4], [4], [0], [6], [7], [4]]
    assertion_message = str(generated_parent_list_vector) + ' should be' + str(expected_parent_list_vector)
    assert generated_parent_list_vector == expected_parent_list_vector, assertion_message


# ::id bolt-eng-DF-170-181103-8889109_0043.4 ::amr-annotator UCO-AMR-05 ::preferred
# ::tok 2012 may be a year of unexpected recovery .
# amr_str = """(y2 / year~e.4
#                   :time-of~e.5 (r / recover-01~e.7
#                         :ARG1-of (e / expect-01 :polarity -~e.6))
#                   :ARG1-of (p / possible-01~e.1)
#                   :domain~e.2 (d / date-entity :year~e.4 2012~e.0))"""
def test_generate_parent_list_vector_with_polarity():
    amr_str = """(y2 / year~e.4
                      :time-of~e.5 (r / recover-01~e.7
                            :ARG1-of (e / expect-01~e.6 :polarity -~e.6))
                      :ARG1-of (p / possible-01~e.1)
                      :domain~e.2 (d / date-entity~e.4 :year~e.4 2012~e.0))"""
    amr: AMR = AMR.parse_string(amr_str)
    identified_concepts = IdentifiedConcepts()
    identified_concepts.create_from_amr('amr_polarity', amr)
    add_false_root(identified_concepts)
    generated_parent_list_vector = generate_parent_list_vector(amr, identified_concepts)
    # 2012 p d y2 - e r
    # 1    2 3 4  5 6 7
    expected_parent_list_vector = [[-1], [3], [4], [4], [0], [6], [7], [4]]
    assertion_message = str(generated_parent_list_vector) + ' should be' + str(expected_parent_list_vector)
    assert generated_parent_list_vector == expected_parent_list_vector, assertion_message


# ::id bolt-eng-DF-170-181103-8889109_0085.26 ::amr-annotator UCO-AMR-05 ::preferred
# ::tok And the only reason they are there is because of insane student loan practices that are identical
# in everyway to the insane mortgage loan practices .
# amr_str = """(a / and~e.0
#                   :op2 (p2 / practice-01~e.13
#                         :ARG1 (l / loan-01~e.12
#                               :ARG2 (p / person~e.11
#                                     :ARG0-of~e.11 (s / study-01~e.11)))
#                         :mod (s2 / sane~e.10 :polarity~e.10 -~e.10)
#                         :ARG1-of (i2 / identical-01~e.16
#                               :ARG2~e.19 (p3 / practice-01~e.24
#                                     :ARG1 (l2 / loan-01~e.23
#                                           :ARG1 (m / mortgage-01~e.22))
#                                     :mod (s3 / sane~e.21 :polarity~e.21 -~e.21))
#                               :manner (w / way
#                                     :mod (e / every)))
#                         :ARG0-of (c2 / cause-01~e.3,8
#                               :ARG1 (b / be-located-at-91~e.5,7
#                                     :ARG1 (t / they~e.4)
#                                     :ARG2 (t2 / there~e.6))
#                               :mod (o / only~e.2))))"""
def test_generate_parent_list_vector_with_2_polarites():
    amr_str = """(a / and~e.0
                      :op2 (p2 / practice-01~e.13
                            :ARG1 (l / loan-01~e.12
                                  :ARG2 (p / person~e.11
                                        :ARG0-of~e.11 (s / study-01~e.11)))
                            :mod (s2 / sane~e.10 :polarity~e.10 -~e.10)
                            :ARG1-of (i2 / identical-01~e.16
                                  :ARG2~e.19 (p3 / practice-01~e.24
                                        :ARG1 (l2 / loan-01~e.23
                                              :ARG1 (m / mortgage-01~e.22))
                                        :mod (s3 / sane~e.21 :polarity~e.21 -~e.21))
                                  :manner (w / way~e.18
                                        :mod (e / every~e.18)))
                            :ARG0-of (c2 / cause-01~e.3,8 
                                  :ARG1 (b / be-located-at-91~e.5,7
                                        :ARG1 (t / they~e.4)
                                        :ARG2 (t2 / there~e.6))
                                  :mod (o / only~e.2))))"""
    amr: AMR = AMR.parse_string(amr_str)
    identified_concepts = IdentifiedConcepts()
    identified_concepts.create_from_amr('amr_2_polarities', amr)
    add_false_root(identified_concepts)
    generated_parent_list_vector = generate_parent_list_vector(amr, identified_concepts)
    # a o c2 t b t2 - s2 s p  l  p2 i2 e  w  -  s3 m  l2 p3
    # 1 2 3  4 5 6  7 8  9 10 11 12 13 14 15 16 17 18 19 20
    expected_parent_list_vector = [[-1], [0], [3], [12], [5], [3], [5], [8],
                                   [12], [10], [11], [12], [1], [12], [15],
                                   [13], [17], [20], [19], [20], [13]]
    assertion_message = str(generated_parent_list_vector) + ' should be' + str(expected_parent_list_vector)
    assert generated_parent_list_vector == expected_parent_list_vector, assertion_message


# ::id bolt12_632_5731.28 ::amr-annotator UCO-AMR-05 ::preferred
# ::tok We have now already received a payment reminder from the hospital .
# amr_str = """(r / receive-01~e.4
#                   :ARG0 (w / we~e.0)
#                   :ARG1 (t / thing~e.7
#                         :ARG0-of~e.7 (r2 / remind-01~e.7
#                               :ARG1 (p / pay-01~e.6
#                                     :ARG0 w)
#                               :ARG2 w))
#                   :ARG2~e.8 (h / hospital~e.10)
#                   :time (n / now~e.2)
#                   :time (a / already~e.3))"""
def test_generate_parent_list_vector_reentrancy():
    amr_str = """(r / receive-01~e.4
                      :ARG0 (w / we~e.0)
                      :ARG1 (t / thing~e.7
                            :ARG0-of~e.7 (r2 / remind-01~e.7
                                  :ARG1 (p / pay-01~e.6
                                        :ARG0 w)
                                  :ARG2 w))
                      :ARG2~e.8 (h / hospital~e.10)
                      :time (n / now~e.2)
                      :time (a / already~e.3))"""
    amr = AMR.parse_string(amr_str)
    identified_concepts = IdentifiedConcepts()
    identified_concepts.create_from_amr('amr_2_polarities', amr)
    add_false_root(identified_concepts)
    generated_parent_list_vector = generate_parent_list_vector(amr, identified_concepts)
    # w n a r p r2 t h
    # 1 2 3 4 5 6  7 8
    expected_parent_list_vector = [ [-1], [5, 6, 4], [4], [4], [0], [6], [7], [4], [4]]
    assertion_message = str(generated_parent_list_vector) + ' should be' + str(expected_parent_list_vector)
    assert generated_parent_list_vector == expected_parent_list_vector, assertion_message


# ::id bolt12_10474_1831.8 ::amr-annotator SDL-AMR-09 ::preferred
# ::tok Am I being foolish in doing this ?
# (f / foolish~e.3 :mode~e.7 interrogative~e.7
#       :domain~e.0,2 (i / i~e.1)
#       :condition~e.4 (d / do-02~e.5
#             :ARG0 i
#             :ARG1 (t / this~e.6)))
def test_generate_parent_list_vector_reentrancy_ex_2():
    amr_str = """(f / foolish~e.3 
                      :mode~e.7 interrogative~e.7
                      :domain~e.0,2 (i / i~e.1)
                      :condition~e.4 (d / do-02~e.5
                                        :ARG0 i
                                        :ARG1 (t / this~e.6)))"""
    amr = AMR.parse_string(amr_str)
    identified_concepts = IdentifiedConcepts()
    identified_concepts.create_from_amr('amr_2_reentrancy', amr)
    add_false_root(identified_concepts)
    generated_parent_list_vector = generate_parent_list_vector(amr, identified_concepts)
    # i f d t interogative
    # 1 2 3 4 5
    expected_parent_list_vector = [ [-1], [3,2], [0], [2], [3], [2]]
    assertion_message = str(generated_parent_list_vector) + ' should be' + str(expected_parent_list_vector)
    assert generated_parent_list_vector == expected_parent_list_vector, assertion_message


# ::id DF-199-193268-677_5527.31 ::amr-annotator SDL-AMR-09 ::preferred
# ::tok I convinced her , but my conviction on this is shallow .
# ::alignments 0-1.1 1-1 2-1.2 2-1.2.r 4-1.3.r 5-1.3.1.1 5-1.3.1.1.r 6-1.3.1 10-1.3
# (c2 / convince-01~e.1
#       :ARG0 (i / i~e.0)
#       :ARG1~e.2 (s / she~e.2)
#       :concession-of~e.4 (s2 / shallow~e.10
#             :ARG1-of (c / conviction-02~e.6
#                   :ARG0~e.5 i~e.5
#                   :ARG2 c2)))
def test_generate_parent_list_vector_reentrancy_ex_3():
    amr_str = """(c2 / convince-01~e.1 
                      :ARG0 (i / i~e.0) 
                      :ARG1~e.2 (s / she~e.2) 
                      :concession-of~e.4 (s2 / shallow~e.10 
                            :ARG1-of (c / conviction-02~e.6 
                                  :ARG0~e.5 i~e.5 
                                  :ARG2 c2)))"""
    amr = AMR.parse_string(amr_str)
    identified_concepts = IdentifiedConcepts()
    identified_concepts.create_from_amr('amr_3_reentrancy', amr)
    add_false_root(identified_concepts)
    generated_parent_list_vector = generate_parent_list_vector(amr, identified_concepts)
    # c2 s i c s2
    # 1  2 3 4 5
    expected_parent_list_vector = [ [-1], [0,4], [1], [4, 1], [5], [1]]
    assertion_message = str(generated_parent_list_vector) + ' should be' + str(expected_parent_list_vector)
    assert generated_parent_list_vector == expected_parent_list_vector, assertion_message


def test_generate_parent_list_vector():
    test_generate_parent_list_vector_ex_1()
    test_generate_parent_list_vector_with_polarity()
    test_generate_parent_list_vector_with_2_polarites()
    test_generate_parent_list_vector_reentrancy()
    test_generate_parent_list_vector_reentrancy_ex_2()
    test_generate_parent_list_vector_reentrancy_ex_3()


if __name__ == "__main__":
    test_generate_parent_list_vector()
    print("Everything in parents_vector_extractor_integration_test passed")
