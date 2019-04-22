from models.amr_graph import AMR
from utilities import generate_action_sequence, generate_custom_amr, generate_amr_with_literals

# amr = AMR.parse_string("""(a3 / and
#       :op1 (s / selfish~e.3
#             :domain~e.1 (p / person~e.0
#                   :quant (a / all~e.2)))
#       :op2 (g / gray-02~e.9
#             :ARG1 (r / reality~e.5)
#             :frequency (o / often~e.8)
#             :mod (a2 / also~e.6)))""")

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

# amr_str = """(c / cause-01~e.0
#       :ARG1 (o / open-01~e.9
#             :ARG1 (a / amusement-park
#                   :name (n / name :op1 "Disneyland"~e.7)
#                   :poss~e.6 (w / we~e.6))
#             :ARG1-of (k / know-01~e.4
#                   :ARG0 (e / everyone~e.3))
#             :time~e.2 (d2 / date-entity :month~e.11 9~e.11
#                   :mod (y / year~e.14
#                         :mod (t / this~e.13)))))"""
# sentence_str = """Because , as everyone knows , our Disneyland will open in September of this year ."""

# amr_str = """(c5 / crown-01~e.6
#       :ARG1 (c / city :wiki "Hong_Kong"
#             :name (n / name :op1 "Hong"~e.0 :op2 "Kong"~e.1))
#       :ARG2~e.7 (l2 / location :wiki -
#             :name (n2 / name :op1 "Hollywood"~e.8 :op2 "of"~e.9 :op3 "the"~e.10 :op4 "East"~e.11))
#       :time (a2 / always~e.3))"""
# sentence = """Hong Kong has always worn the crown of Hollywood of the East ."""

# amr_str = """(c5 / crown-01~e.6
#       :ARG1 (c / city :wiki "Hong_Kong"
#             :name (n / name :op1 "Hong"~e.0 :op2 "Kong"~e.1))
#       :ARG2~e.7 (l2 / location :wiki -
#             :name (n2 / name :op1 "Hollywood"~e.8 :op2 "of"~e.9 :op3 "the"~e.10 :op4 "East"~e.11))
#       :time (a2 / always~e.3))"""
#
# sentence = "Hong Kong has always worn the crown of Hollywood of the East ."

# amr_str = """(r / recommend-01~e.1
#                 :ARG1 (a / advocate-01~e.4
#                     :ARG1 (i / it~e.0)
#                     :manner~e.2 (v / vigorous~e.3)))"""
# sentence = """It should be vigorously advocated ."""

# amr_str = """(b / become-01~e.6
#       :ARG1 (a / area~e.4
#             :mod (t / this~e.3))
#       :ARG2 (z / zone~e.9
#             :ARG1-of (p / prohibit-01~e.8)
#             :part-of~e.10 (c / city :wiki "Hong_Kong"
#                   :name (n / name :op1 "Hong"~e.11 :op2 "Kong"~e.12)))
#       :time (s / since~e.0
#             :op1 (t2 / then~e.1)))"""
# sentence = """Since then , this area has become a prohibited zone in Hong Kong ."""

# amr_str = """(p / person :wiki "Goddess"
#       :name (n / name :op1 "Goddess"~e.3)
#       :domain~e.1 (s / she~e.0)
#       :poss~e.2 (i / i~e.2)
#       :mod (a / ah~e.5))"""
# sentence = """She is my Goddess , ah ."""

sentence = """The official alleged Karzai was reluctant to move against big drug lords in Karzai 's political power base in the south of Afghanistan where most opium is produced ."""
amr_str = """(a / allege-01~e.2
      :ARG0 (p5 / person
            :ARG0-of (h / have-org-role-91
                  :ARG2 (o / official~e.1)))
      :ARG1 (r / reluctant~e.5
            :topic (m / move-04~e.7
                  :ARG0 p
                  :ARG1 (l / lord~e.11
                        :mod (d / drug~e.10)
                        :mod (b / big~e.9))
                  :location~e.12 (b2 / base~e.17
                        :mod (p2 / power~e.16)
                        :mod (p3 / politics~e.15)
                        :poss p~e.13,14
                        :location~e.18 (s2 / south~e.20
                              :part-of~e.21 (c / country :wiki "Afghanistan"
                                    :name (n / name :op1 "Afghanistan"~e.22))
                              :location-of~e.23 (p4 / produce-01~e.27
                                    :ARG1 (o2 / opium~e.25
                                          :quant (m2 / most~e.24))))))
            :domain~e.4 (p / person :wiki "Hamid_Karzai"
                  :name (n3 / name :op1 "Karzai"~e.3))))"""


def test_literals(amr_str, sentence):
    (new_amr, new_sentence, named_entities) = generate_amr_with_literals(amr_str, sentence)
    custom_amr = generate_custom_amr(new_amr)
    seq = generate_action_sequence(custom_amr, new_sentence)
    print seq


def test_amr(amr_str, sentence):
    amr = AMR.parse_string(amr_str)
    custom_amr = generate_custom_amr(amr)
    seq = generate_action_sequence(custom_amr, sentence)
    print seq


test_literals(amr_str, sentence)
