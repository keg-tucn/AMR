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
def test_create_from_amr_example_1():
    amr_str = """(r / recommend-01~e.1
                    :ARG1 (a / advocate-01~e.4
                        :ARG1 (i / it~e.0)
                        :manner~e.2 (v / vigorous~e.3)))"""
    amr = AMR.parse_string(amr_str)
    generated_concepts = IdentifiedConcepts()
    generated_concepts.create_from_amr('amr_id_1', amr)
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
def test_create_from_amr_example_2():
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
    generated_concepts.create_from_amr('amr_id_2', amr)
    expected_concepts = IdentifiedConcepts()
    expected_concepts.amr_id = 'amr_id_2'
    expected_concepts.ordered_concepts = [Concept('a', 'and'),
                                          Concept('-', '-', 0),
                                          Concept('a4', 'any'),
                                          Concept('p2', 'promise-01'),
                                          Concept('h', 'he'),
                                          Concept('p', 'possible-01'),
                                          Concept('a2', 'actual-02'),
                                          Concept('a3', 'avoid-01'),
                                          Concept('c', 'censure-01')
                                          ]
    assert_identified_concepts(expected_concepts, generated_concepts)


# """(d / difficult~e.5
#           :domain~e.4 (r / reach-01~e.7
#                 :ARG1 (c / consensus~e.0
#                       :topic~e.1 (c2 / country :wiki "India"
#                             :name (n / name :op1 "India"~e.2)))
#                 :time~e.8 (m / meet-03~e.11
#                       :ARG0 (o / organization :wiki "Nuclear_Suppliers_Group"
#                             :name (n2 / name :op1 "NSG"~e.10))
#                       :time~e.12 (d2 / date-entity :year 2007~e.14 :month~e.13 11~e.13))))"""
def test_create_from_amr_example_3():
    amr_str = """(d / difficult~e.5
          :domain~e.4 (r / reach-01~e.7
                :ARG1 (c / consensus~e.0
                      :topic~e.1 (c2 / country :wiki "India"
                            :name (n / name :op1 "India"~e.2)))
                :time~e.8 (m / meet-03~e.11
                      :ARG0 (o / organization :wiki "Nubolt12_632_6421.19clear_Suppliers_Group"
                            :name (n2 / name :op1 "NSG"~e.10))
                      :time~e.12 (d2 / date-entity :year 2007~e.14 :month~e.13 11~e.13))))"""
    amr = AMR.parse_string(amr_str)
    generated_concepts = IdentifiedConcepts()
    generated_concepts.create_from_amr('amr_id_3', amr)
    expected_concepts = IdentifiedConcepts()
    expected_concepts.amr_id = 'amr_id_3'
    # return None as not all concepts are aligned + unalignment tolerance is default (0)
    expected_concepts.ordered_concepts = None
    assert_identified_concepts(expected_concepts, generated_concepts)


# ::id  ::amr-annotator SDL-AMR-09 ::preferred
# ::tok Finally , the contradictions are bound to intensify , making the situation out of hand .
# ::alignments 0-1.1 0-1.1.r 3-1.2 5-1.4 7-1 9-1.3 11-1.3.1.2
# (i / intensify-01~e.7 :li~e.0 -1~e.0
#       :ARG1 (c / contradiction~e.3)
#       :ARG0-of (m / make-02~e.9
#             :ARG1 (c2 / control-01 :polarity -
#                   :ARG1 (s / situation~e.11)))
#       :ARG1-of (b / bind-02~e.5))
def test_create_from_amr_example_4():
    amr_str = """(i / intensify-01~e.7 :li~e.0 -1~e.0 
                    :ARG1 (c / contradiction~e.3) 
                    :ARG0-of (m / make-02~e.9 
                        :ARG1 (c2 / control-01~e.12,13,14 :polarity - 
                              :ARG1 (s / situation~e.11))) 
                    :ARG1-of (b / bind-02~e.5))"""
    amr = AMR.parse_string(amr_str)
    generated_concepts = IdentifiedConcepts()
    generated_concepts.create_from_amr('amr_id_3', amr)
    expected_concepts = IdentifiedConcepts()
    expected_concepts.amr_id = 'amr_id_3'
    # return None as not all concepts are aligned + unalignment tolerance is default (0)
    expected_concepts.ordered_concepts = None
    assert_identified_concepts(expected_concepts, generated_concepts)

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
def test_create_from_amr_example_reentrancy():
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
    generated_concepts = IdentifiedConcepts()
    generated_concepts.create_from_amr('amr_id_reentrancy', amr)
    expected_concepts = IdentifiedConcepts()
    expected_concepts.amr_id = 'amr_id_reentrancy'
    expected_concepts.ordered_concepts = [Concept('w', 'we'),
                                          Concept('n', 'now'),
                                          Concept('a', 'already'),
                                          Concept('r', 'receive-01'),
                                          Concept('p', 'pay-01'),
                                          Concept('r2', 'remind-01'),
                                          Concept('t', 'thing'),
                                          Concept('h', 'hospital')]
    assert_identified_concepts(expected_concepts, generated_concepts)


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
def test__create_from_amr_with_2_polarites():
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
    generated_concepts = IdentifiedConcepts()
    generated_concepts.create_from_amr('amr_2_polarities', amr)
    expected_concepts = IdentifiedConcepts()
    expected_concepts.amr_id = 'amr_2_polarities'
    expected_concepts.ordered_concepts = [Concept('a', 'and'),
                                          Concept('o', 'only'),
                                          Concept('c2', 'cause-01'),
                                          Concept('t', 'they'),
                                          Concept('b', 'be-located-at-91'),
                                          Concept('t2', 'there'),
                                          Concept('-', '-', 0),
                                          Concept('s2', 'sane'),
                                          Concept('s', 'study-01'),
                                          Concept('p', 'person'),
                                          Concept('l', 'loan-01'),
                                          Concept('p2', 'practice-01'),
                                          Concept('i2', 'identical-01'),
                                          Concept('e', 'every'),
                                          Concept('w', 'way'),
                                          Concept('-', '-', 1),
                                          Concept('s3', 'sane'),
                                          Concept('m', 'mortgage-01'),
                                          Concept('l2', 'loan-01'),
                                          Concept('p3', 'practice-01')]
    assert_identified_concepts(expected_concepts, generated_concepts)


def test_create_from_amr():
    test_create_from_amr_example_1()
    test_create_from_amr_example_2()
    test_create_from_amr_example_3()
    test_create_from_amr_example_4()
    test_create_from_amr_example_reentrancy()
    test__create_from_amr_with_2_polarites()


if __name__ == "__main__":
    test_create_from_amr()
    print("Everything in concept_integration_test passed")
