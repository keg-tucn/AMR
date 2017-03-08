from AMRGraph import AMR
from AMRData import CustomizedAMR
import ActionSequenceGenerator

# amr = AMR.parse_string("""(a3 / and
#       :op1 (s / selfish~e.3
#             :domain~e.1 (p / person~e.0
#                   :quant (a / all~e.2)))
#       :op2 (g / gray-02~e.9
#             :ARG1 (r / reality~e.5)
#             :frequency (o / often~e.8)
#             :mod (a2 / also~e.6)))""")

# amr = AMR.parse_string("""(r / recommend-01~e.1
#       :ARG1 (a / advocate-01~e.4
#             :ARG1 (i / it~e.0)
#             :manner~e.2 (v / vigorous~e.3)))""")

# amr = AMR.parse_string("""(m / multi-sentence
#      :snt1 (e / exemplify-01~e.1
#            :ARG0 (p / person :wiki "Li_Yinhe"
#                  :name (n / name :op1 "Li"~e.14 :op2 "Yinhe"~e.15)
#                  :mod (s / sexologist~e.7
#                        :ARG1-of (c / call-01~e.5
#                              :mod (s3 / so~e.3)))
#                  :mod (a2 / activist~e.12
#                        :mod (s2 / social~e.11)
#                        :ARG1-of c))
#            :mod (a3 / another~e.0))
#      :snt2 (r / resemble-01~e.23 :polarity~e.22 -~e.22
#            :ARG1 (s4 / stuff~e.20
#                  :poss~e.19 (s6 / she~e.19))
#            :ARG2~e.24 (t / thing~e.25
#                  :example~e.26 (t2 / treatise~e.28 :wiki -
#                        :name (n3 / name :op1 "Haite"~e.30 :op2 "Sexology"~e.31 :op3 "Report"~e.32)))
#            :li (x / 1~e.17))
#      :snt3 (t3 / think-01~e.37,40 :polarity~e.38 -~e.38
#            :ARG0 (s7 / she~e.36)
#            :ARG1-of (d / deep-02~e.39)
#            :li (x2 / 2~e.34))
#      :snt4 (c2 / contribute-01~e.49 :polarity~e.47 -~e.47
#            :ARG0 (s8 / she~e.45)
#            :ARG1-of (s5 / significant-02~e.48)
#            :li (x3 / 3~e.43)))""")

#
# amr = AMR.parse_string("""(m / multi-sentence
#       :snt1 (r / return-01~e.3
#             :ARG1 (i / i~e.0)
#             :ARG4~e.4 (c / country~e.6
#                   :poss~e.5 i~e.5)
#             :time~e.7 (d / date-entity :month~e.8 9~e.8))
#       :snt2 (p2 / place~e.14 :quant 2~e.13
#             :ARG4-of (g / go-02~e.17
#                   :ARG0 (i2 / i~e.15)
#                   :time (n / now~e.10)
#                   :ARG1-of (p / possible-01~e.16)))
#       :snt3 (p3 / possible-01~e.20 :mode~e.31 interrogative~e.31
#             :ARG1 (h / help-01~e.22
#                   :ARG0 (e / everyone~e.21)
#                   :ARG1 (s / see-01~e.24
#                         :ARG0 i3~e.23
#                         :ARG1 (t / thing~e.28
#                               :ARG1-of~e.28 (s2 / suitable-04~e.28
#                                     :ARG2 i3)
#                               :degree (m2 / more~e.27)))
#                   :ARG2~e.29 (i3 / i~e.30))))""")

# amr = AMR.parse_string("""(s2 / seem-01~e.1
#       :ARG1~e.3 (h / have-03~e.8
#             :ARG0 (w / we~e.7)
#             :ARG1 (s / scheme~e.10
#                   :mod (p / plan-01~e.15
#                         :ARG1 (r / renovate-01~e.14)
#                         :ARG1-of (m / major-02~e.13))
#                   :purpose (f / future~e.5))))""")

amr = AMR.parse_string("""(l / look-02~e.1
      :ARG1~e.2 (b / bring-01~e.6
            :ARG0 (w / we~e.3)
            :ARG1~e.7 (w2 / whale~e.8)
            :mod (a / also~e.5)))""")


def pretty_print(amr):
    for k in amr.keys():
        print "Key: %s" % (k)
        list = amr[k]
        if len(list) == 0:
            print "Leaf"
        for rel in list:
            print "%s -> %s" % (rel, list[rel][0])
        print ""


print "\nMappings between node variables and their corresponding concepts.\n"
print amr.node_to_concepts

print "\nMappings between nodes and all the aligned tokens: If the nodes don't have" \
      "a variable (polarity, literals, quantities, interrogatives), they specify both the aligned tokens " \
      "and the parent in order to uniquely identify them\n"
print amr.node_to_tokens

print "\nMappings between relations and tokens. Uniquely identified by also specifying the parent of that relation.\n"
# TODO: since the clean-up of parents which are not actual variables is done at the final, we might end up
# having parents such as 9~e.15 for the relations. However, as I've seen so far, these kind of nodes are usually leaves
# so hopefully we won't have this problem
print amr.relation_to_tokens

print "\nMappings from a node to each child, along with the relation between them.\n"
pretty_print(amr)

print "\nAll the nodes in the amr should appear here.\n"
print amr.keys()

print "\nCreating custom AMR.\n"
custom_AMR = CustomizedAMR()
custom_AMR.create_custom_AMR(amr)
print "\nCustom AMR token to concepts dict\n"
print custom_AMR.tokens_to_concepts_dict
print "\nCustom AMR relations dict\n"
print custom_AMR.relations_dict
print "\nCustom AMR parent dict\n"
print custom_AMR.parent_dict

print(ActionSequenceGenerator.generate_action_sequence(custom_AMR,
                                                     "It looks like we will also bring in whales ."))

#['DN', 'SH_look-02', 'DN', 'SH_we', 'DN', 'SH_also', 'SH_bring-01', 'RL_mod', 'RL_ARG0', 'DN', 'SH_whale', 'RR_ARG1', 'RR_ARG1', 'DN']