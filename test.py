from AMRGraph import AMR

amr = AMR.parse_string("""(a3 / and
      :op1 (s / selfish~e.3
            :domain~e.1 (p / person~e.0
                  :quant (a / all~e.2)))
      :op2 (g / gray-02~e.9
            :ARG1 (r / reality~e.5)
            :frequency (o / often~e.8)
            :mod (a2 / also~e.6)))""")

amr = AMR.parse_string("""(m / multi-sentence
     :snt1 (e / exemplify-01~e.1
           :ARG0 (p / person :wiki "Li_Yinhe"
                 :name (n / name :op1 "Li"~e.14 :op2 "Yinhe"~e.15)
                 :mod (s / sexologist~e.7
                       :ARG1-of (c / call-01~e.5
                             :mod (s3 / so~e.3)))
                 :mod (a2 / activist~e.12
                       :mod (s2 / social~e.11)
                       :ARG1-of c))
           :mod (a3 / another~e.0))
     :snt2 (r / resemble-01~e.23 :polarity~e.22 -~e.22
           :ARG1 (s4 / stuff~e.20
                 :poss~e.19 (s6 / she~e.19))
           :ARG2~e.24 (t / thing~e.25
                 :example~e.26 (t2 / treatise~e.28 :wiki -
                       :name (n3 / name :op1 "Haite"~e.30 :op2 "Sexology"~e.31 :op3 "Report"~e.32)))
           :li (x / 1~e.17))
     :snt3 (t3 / think-01~e.37,40 :polarity~e.38 -~e.38
           :ARG0 (s7 / she~e.36)
           :ARG1-of (d / deep-02~e.39)
           :li (x2 / 2~e.34))
     :snt4 (c2 / contribute-01~e.49 :polarity~e.47 -~e.47
           :ARG0 (s8 / she~e.45)
           :ARG1-of (s5 / significant-02~e.48)
           :li (x3 / 3~e.43)))""")
print amr.node_to_concepts
print amr.node_to_tokens
print amr.external_nodes
print amr.items()
