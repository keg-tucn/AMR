from models.amr_data import CustomizedAMR
from models.amr_graph import AMR
from preprocessing.preprocessing_steps import apply_preprocessing_steps_on_instance, NamedEntitiesPreprocessingStep, \
    DateEntitiesPreprocessingStep, TemporalQuantitiesPreprocessingStep, QuantitiesPreprocessingStep, \
    HaveOrgPreprocessingStep, apply_preprocessing_steps_on_amr_list


# ::id wb.eng_0009.86 ::amr-annotator SDL-AMR-09 ::preferred
# ::tok The pop star , who is said to be $ 240 million in
# debt , had paid six figures for a ritual cleansing using sheep blood to another voodoo doctor and a mysterious
# Egyptian woman named Samia , who came to him with a letter of greeting from a high - ranking Saudi prince ,
# purportedly Nawaf Bin Abdulaziz Al - Saud , now the chief of intelligence of Saudi Arabia ...
def test_apply_preprocessing_steps_on_instance_quantity_after_named_entitity():
    amr_str = """(p / pay-01~e.16 
      :ARG0 (p2 / person 
            :mod (s / star~e.2 
                  :mod (p3 / pop~e.1)) 
            :mod (d3 / debt~e.13 
                  :consist-of (m / monetary-quantity :quant 240000000~e.10,11 
                        :unit (d2 / dollar~e.9)) 
                  :ARG1-of (s2 / say-01~e.6))) 
      :ARG1 (m2 / monetary-quantity :quant 6~e.17 
            :unit (f / figure~e.18)) 
      :ARG2~e.26 (a / and~e.30 
            :op1 (d / doctor~e.29 
                  :mod (v / voodoo~e.28) 
                  :mod (a2 / another~e.27)) 
            :op2 (w / woman~e.34 :wiki - 
                  :name (n / name~e.35 :op1 "Samia"~e.36) 
                  :mod (m3 / mystery~e.32) 
                  :mod (c2 / country :wiki "Egypt" 
                        :name (n2 / name~e.35 :op1 "Egypt"~e.33)) 
                  :ARG1-of (c3 / come-01~e.39 
                        :ARG4~e.40 p2~e.41 
                        :accompanier~e.42 (l / letter~e.44 
                              :mod~e.45 (g / greet-01~e.46 
                                    :ARG0~e.47 (p7 / person 
                                          :ARG0-of (h2 / have-org-role-91 
                                                :ARG1 c6 
                                                :ARG2 (p4 / prince~e.53 
                                                      :ARG1-of (r2 / rank-01~e.51 
                                                            :ARG1-of (h / high-02~e.49)))) 
                                          :ARG0-of (p5 / purport-01 
                                                :ARG1 (p6 / person :wiki "Mohammed_bin_Nawwaf_bin_Abdulaziz" 
                                                      :name (n4 / name~e.35 :op1 "Nawaf"~e.56 :op2 "Bin"~e.57 :op3 "Abdulaziz"~e.58 :op4 "Al"~e.59 :op5 "Saud"~e.61) 
                                                      :ARG0-of (h3 / have-org-role-91~e.68 
                                                            :ARG1 (c6 / country :wiki "Saudi_Arabia" 
                                                                  :name (n5 / name~e.35 :op1 "Saudi"~e.69 :op2 "Arabia"~e.70)) 
                                                            :ARG2 (c5 / chief~e.65 
                                                                  :topic~e.66 (i / intelligence~e.67)) 
                                                            :time (n6 / now~e.63)))))))))) 
      :ARG3~e.19 (c / cleanse-01~e.22 
            :manner (r / ritual~e.21) 
            :ARG0-of (u / use-01~e.23 
                  :ARG1 (b / blood~e.25 
                        :mod (s3 / sheep~e.24)))))"""
    sentence = """The pop star , who is said to be $ 240 million in debt , had paid six figures for a ritual 
    cleansing using sheep blood to another voodoo doctor and a mysterious Egyptian woman named Samia , who came to 
    him with a letter of greeting from a high - ranking Saudi prince , purportedly Nawaf Bin Abdulaziz Al - Saud , 
    now the chief of intelligence of Saudi Arabia ... """
    amr = AMR.parse_string(amr_str)
    print("Before preprocessing ")
    print(amr)
    new_amr, sentence, _ = apply_preprocessing_steps_on_instance(amr, sentence,
                                                                 [
                                                                     NamedEntitiesPreprocessingStep()
                                                                 ])
    print("After NE preprocessing ")
    print(new_amr)
    new_amr2, sentence, _ = apply_preprocessing_steps_on_instance(new_amr, sentence,
                                                                  [
                                                                      QuantitiesPreprocessingStep()
                                                                  ])
    print("After Q preprocessing ")
    print(new_amr2)
    custom_amr = CustomizedAMR()
    # custom_amr.create_custom_AMR(new_amr_list[0][1])
    custom_amr.create_custom_AMR(new_amr2)
    # TODO: finish test


def test_apply_preprocessing_steps_on_instance():
    test_apply_preprocessing_steps_on_instance_quantity_after_named_entitity()


if __name__ == "__main__":
    test_apply_preprocessing_steps_on_instance()
    print("Everything in preprocessing_steps_integration_test passed")
