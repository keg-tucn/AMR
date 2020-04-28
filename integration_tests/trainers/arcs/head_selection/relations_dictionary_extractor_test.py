from models.amr_data import CustomizedAMR
from models.amr_graph import AMR
from trainers.arcs.head_selection.relations_dictionary_extractor import extract_relation_counts_dict


# id bolt12_12260_1905.7 ::amr-annotator SDL-AMR-09
# sentence1 = """How China University of Mining and Technology responds to Jiangsu
# mainstream media 's positive and active appraisal of Wang Peirong ;
# how well they treat Wang Peirong ; and how they protect Wang Peirong 's legal rights ,
# are unavoidable questions placed in front of the Mining University 's decision makers ."""
# id DF-200-192451-579_6283.18
# sentence2 = """And that 's a bad thing why ?"""
def test_extract_relation_counts_dict():
    amr_str_1 = """(t / thing~e.40 
          :ARG1-of~e.40 (q / question-01~e.40) 
          :ARG1-of (p / place-01~e.41 
                :ARG2 (i / in-front-of~e.42,43,44 
                      :op1 (p7 / person~e.49 
                            :ARG0-of~e.49 (d / decide-01~e.49) 
                            :ARG0-of (h / have-org-role-91~e.48 
                                  :ARG1 u)))) 
          :domain~e.34,38,48 (a / and~e.28 
                :op1 (t3 / thing 
                      :manner-of~e.0 (r / respond-01~e.7 
                            :ARG0 (u / university~e.47 :wiki "China_University_of_Mining_and_Technology" 
                                  :name (n / name :op1 "China"~e.1 :op2 "University"~e.2 :op3 "of"~e.3 :op4 "Mining"~e.4 :op5 "and"~e.5 :op6 "Technology"~e.6)) 
                            :ARG1~e.8 (a5 / appraise-02~e.16 
                                  :ARG0~e.12 (m / media~e.11 
                                        :ARG1-of (m2 / mainstream-02~e.10) 
                                        :location (p3 / province :wiki "Jiangsu" 
                                              :name (n2 / name :op1 "Jiangsu"~e.9))) 
                                  :ARG1~e.17 (p4 / person :wiki - 
                                        :name (n3 / name :op1 "Wang"~e.18 :op2 "Peirong"~e.19)) 
                                  :manner~e.21 (p5 / positive~e.13) 
                                  :manner~e.21 (a6 / active~e.15)))) 
                :op2 (t6 / thing 
                      :degree-of (w / well-09~e.22 
                            :manner-of (t2 / treat-01~e.24 
                                  :ARG0 u 
                                  :ARG1 p4~e.25,26))) 
                :op3 (t5 / thing 
                      :manner-of~e.29 (p2 / protect-01~e.31 
                            :ARG0 u~e.30 
                            :ARG1 (r2 / right-05~e.36 
                                  :ARG1~e.34 p4~e.32,33 
                                  :ARG1-of (l / legal-02~e.35))))) 
          :ARG1-of (a7 / avoid-01 
                :ARG0 p7 
                :ARG1-of (p6 / possible-01 :polarity -~e.39)))"""
    amr1 = AMR.parse_string(amr_str_1)
    custom_amr_1 = CustomizedAMR()
    custom_amr_1.create_custom_AMR(amr1)
    amr_str_2 = """(a3 / and~e.0 
      :op1 (t / thing~e.5 
            :ARG1-of (b / bad-07~e.4) 
            :domain~e.2 (t2 / that~e.1) 
            :ARG1-of (c / cause-01~e.6 
                  :ARG0~e.6 (a / amr-unknown~e.6)) 
            :ARG1-of (c2 / cause-01~e.6 
                  :ARG0~e.6 (a2 / amr-unknown~e.6))))"""
    amr2 = AMR.parse_string(amr_str_2)
    custom_amr_2 = CustomizedAMR()
    custom_amr_2.create_custom_AMR(amr2)
    custom_amrs = [custom_amr_1, custom_amr_2]
    generated_relations_counts = extract_relation_counts_dict(custom_amrs)
    expected_relations_counts = {('university', 'China_University_of_Mining_and_Technology'): {'wiki': 1},
                                 ('university', 'name'): {'name': 1},
                                 ('have-org-role-91', 'university'): {'ARG1': 1},
                                 ('person', 'decide-01'): {'ARG0-of': 1},
                                 ('person', 'have-org-role-91'): {'ARG0-of': 1},
                                 ('in-front-of', 'person'): {'op1': 1},
                                 ('place-01', 'in-front-of'): {'ARG2': 1},
                                 ('name', 'China'): {'op1': 1},
                                 ('name', 'University'): {'op2': 1},
                                 ('name', 'of'): {'op3': 1},
                                 ('name', 'Mining'): {'op4': 1},
                                 ('name', 'and'): {'op5': 1},
                                 ('name', 'Technology'): {'op6': 1},
                                 ('name', 'Jiangsu'): {'op1': 1},
                                 ('province', 'Jiangsu'): {'wiki': 1},
                                 ('province', 'name'): {'name': 1},
                                 ('media', 'mainstream-02'): {'ARG1-of': 1},
                                 ('media', 'province'): {'location': 1},
                                 ('name', 'Wang'): {'op1': 1},
                                 ('name', 'Peirong'): {'op2': 1},
                                 ('person', '-'): {'wiki': 1},
                                 ('person', 'name'): {'name': 1},
                                 ('appraise-02', 'media'): {'ARG0': 1},
                                 ('appraise-02', 'person'): {'ARG1': 1},
                                 ('appraise-02', 'positive'): {'manner': 1},
                                 ('appraise-02', 'active'): {'manner': 1},
                                 ('respond-01', 'university'): {'ARG0': 1},
                                 ('respond-01', 'appraise-02'): {'ARG1': 1},
                                 ('thing', 'respond-01'): {'manner-of': 1},
                                 ('treat-01', 'university'): {'ARG0': 1},
                                 ('treat-01', 'person'): {'ARG1': 1},
                                 ('well-09', 'treat-01'): {'manner-of': 1},
                                 ('thing', 'well-09'): {'degree-of': 1},
                                 ('right-05', 'person'): {'ARG1': 1},
                                 ('right-05', 'legal-02'): {'ARG1-of': 1},
                                 ('protect-01', 'university'): {'ARG0': 1},
                                 ('protect-01', 'right-05'): {'ARG1': 1},
                                 ('thing', 'protect-01'): {'manner-of': 1},
                                 ('and', 'thing'): {'op1': 2, 'op2': 1, 'op3': 1},
                                 ('possible-01', '-'): {'polarity': 1},
                                 ('avoid-01', 'person'): {'ARG0': 1},
                                 ('avoid-01', 'possible-01'): {'ARG1-of': 1},
                                 ('thing', 'question-01'): {'ARG1-of': 1},
                                 ('thing', 'place-01'): {'ARG1-of': 1},
                                 ('thing', 'avoid-01'): {'ARG1-of': 1},
                                 ('thing', 'and'): {'domain': 1},
                                 ('cause-01', 'amr-unknown'): {'ARG0': 2},
                                 ('thing', 'bad-07'): {'ARG1-of': 1},
                                 ('thing', 'cause-01'): {'ARG1-of': 2},
                                 ('thing', 'that'): {'domain': 1}}
    assert expected_relations_counts == generated_relations_counts


if __name__ == "__main__":
    test_extract_relation_counts_dict()
    print("Everything in relations_dictionary_extractor passed")
