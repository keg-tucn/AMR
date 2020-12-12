from models.amr_graph import AMR
from models.concept import IdentifiedConcepts
from models.node import Node
from pre_post_processing.standford_pre_post_processing import train_pre_processing, post_processing_on_parent_vector
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.parents_vector_extractor import \
    generate_parent_list_vector
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.arcs_trainer_util import \
    generate_amr_node_for_vector_of_parents, calculate_smatch
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.training_arcs_data_extractor import add_false_root


# ::tok Some propaganda activities of ZF have soon become viewed as jokes by the people .
# (b / become-01~e.7
#       :ARG1 a
#       :ARG2 (v / view-02~e.8
#             :ARG0~e.11 (p2 / person~e.13)
#             :ARG1 (a / activity-06~e.2
#                   :ARG0~e.3 (o / organization :wiki -
#                         :name (n / name :op1 "ZF"~e.4))
#                   :ARG1 (p / propaganda~e.1)
#                   :quant (s / some~e.0))
#             :ARG2~e.9 (t / thing~e.10
#                   :ARG2-of~e.10 (j / joke-01~e.10)))
#       :time~e.9 (s2 / soon~e.6))
def test_pre_and_post_processing_for_organization():
    sentence = 'Some propaganda activities of ZF have soon become viewed as jokes by the people .'
    amr_str = """(b / become-01~e.7
      :ARG1 a
      :ARG2 (v / view-02~e.8
            :ARG0~e.11 (p2 / person~e.13)
            :ARG1 (a / activity-06~e.2
                  :ARG0~e.3 (o / organization :wiki ZF
                        :name (n / name :op1 "ZF"~e.4))
                  :ARG1 (p / propaganda~e.1)
                  :quant (s / some~e.0))
            :ARG2~e.9 (t / thing~e.10
                  :ARG2-of~e.10 (j / joke-01~e.10)))
      :time~e.9 (s2 / soon~e.6))"""
    amr: AMR = AMR.parse_string(amr_str)
    amr, new_sentence, metadata = train_pre_processing(amr, sentence)
    identified_concepts = IdentifiedConcepts()
    identified_concepts.create_from_amr('', amr)
    add_false_root(identified_concepts)
    vector_of_parents = generate_parent_list_vector(amr, identified_concepts)
    post_processing_on_parent_vector(identified_concepts, vector_of_parents, new_sentence, metadata)
    relations_dict = {('become-01', 'activity-06'): 'ARG1', ('become-01', 'view-02'): 'ARG2',
                      ('become-01', 'soon'): 'time',
                      ('view-02', 'person'): 'ARG0', ('view-02', 'activity-06'): 'ARG1', ('view-02', 'thing'): 'ARG2',
                      ('activity-06', 'organization'): 'ARG0', ('activity-06', 'propaganda'): 'ARG1',
                      ('activity-06', 'some'): 'quant',
                      ('organization', 'ZF'): 'wiki', ('organization', 'name'): 'name',
                      ('name', 'ZF'): 'op1', ('thing', 'joke-01'): 'ARG2-of'}
    amr_node: Node = generate_amr_node_for_vector_of_parents(identified_concepts, vector_of_parents, relations_dict)
    generated_amr_str = amr_node.amr_print_with_reentrancy()
    smatch = calculate_smatch(generated_amr_str, amr_str)
    assert smatch == 1


# ::id DF-170-181103-888_4397.1 ::amr-annotator LDC-AMR-14 ::preferred
# ::tok It is Santorum that is the by far major nonRomney candidate and Newt would appear to be the spoiler .
# (a / and~e.11
#       :op1 (c / candidate~e.10
#             :ARG1-of (m / major-02~e.8
#                   :degree (b / by-far~e.6,7))
#             :mod (p3 / person :polarity - :wiki "Mitt_Romney"
#                   :name (n2 / name :op1 "Romney"))
#             :domain~e.1,4 (p2 / person :wiki "Rick_Santorum"
#                   :name (n / name :op1 "Santorum"~e.2)))
#       :op2 (a2 / appear-02~e.14
#             :ARG1 (s / spoil-01
#                   :ARG0 (p4 / person :wiki "Newt_Gingrich"
#                         :name (n3 / name :op1 "Newt"~e.12)))))
def test_pre_and_post_processing_eg_2():
    sentence = 'It is Santorum that is the by far major nonRomney candidate and Newt would appear to be the spoiler .'
    amr_str = """(a / and~e.11
      :op1 (c / candidate~e.10
            :ARG1-of (m / major-02~e.8
                  :degree (b / by-far~e.6,7))
            :mod (p3 / person~e.9 :polarity -~e.9 :wiki "Mitt_Romney"~e.9
                  :name (n2 / name~e.9 :op1 "Romney"~e.9))
            :domain~e.1,4 (p2 / person :wiki "Rick_Santorum"
                  :name (n / name :op1 "Santorum"~e.2)))
      :op2 (a2 / appear-02~e.14
            :ARG1 (s / spoil-01~e.18
                  :ARG0 (p4 / person :wiki "Newt_Gingrich"
                        :name (n3 / name :op1 "Newt"~e.12)))))"""
    amr: AMR = AMR.parse_string(amr_str)
    amr, new_sentence, metadata = train_pre_processing(amr, sentence)
    identified_concepts = IdentifiedConcepts()
    identified_concepts.create_from_amr('', amr)
    add_false_root(identified_concepts)
    vector_of_parents = generate_parent_list_vector(amr, identified_concepts)
    post_processing_on_parent_vector(identified_concepts, vector_of_parents, new_sentence, metadata)
    relations_dict = {('and', 'candidate'): 'op1', ('and', 'appear-02'): 'op2',
                      ('candidate', 'major-02'): 'ARG1-of', ('candidate', 'person'): 'mod',
                      ('major-02', 'by-far'): 'degree',
                      ('person', '-'): 'polarity', ('person', 'Mitt_Romney'): 'wiki', ('person', 'name'): 'name',
                      ('person', 'Santorum'): 'wiki',
                      ('name', 'Romney'): 'op1', ('name', 'Santorum'): 'op1',
                      ('appear-02', 'spoil-01'): 'ARG1', ('spoil-01', 'person'): 'ARG0',
                      ('person', 'Newt'): 'wiki', ('name', 'Newt'): 'op1'
                      }
    amr_node: Node = generate_amr_node_for_vector_of_parents(identified_concepts, vector_of_parents, relations_dict)
    generated_amr_str = amr_node.amr_print_with_reentrancy()
    expected_amr_str = """(a / and~e.11
      :op1 (c / candidate~e.10
            :ARG1-of (m / major-02~e.8
                  :degree (b / by-far~e.6,7))
            :mod (p3 / person~e.9 :polarity -~e.9 :wiki "Mitt_Romney"~e.9
                  :name (n2 / name~e.9 :op1 "Romney"~e.9))
            :mod~e.1,4 (p2 / person :wiki "Santorum"
                  :name (n / name :op1 "Santorum"~e.2)))
      :op2 (a2 / appear-02~e.14
            :ARG1 (s / spoil-01~e.18
                  :ARG0 (p4 / person :wiki "Newt"
                        :name (n3 / name :op1 "Newt"~e.12)))))"""

    smatch = calculate_smatch(generated_amr_str, expected_amr_str)
    assert smatch == 1


if __name__ == "__main__":
    test_pre_and_post_processing_for_organization()
    test_pre_and_post_processing_eg_2()
