from AMRGraph import AMR

amr = AMR.parse_string("""(a3 / and
      :op1 (s / selfish~e.3
            :domain~e.1 (p / person~e.0
                  :quant (a / all~e.2)))
      :op2 (g / gray-02~e.9
            :ARG1 (r / reality~e.5)
            :frequency (o / often~e.8)
            :mod (a2 / also~e.6)))""")

print amr.node_to_concepts
print amr.node_to_tokens
print amr.external_nodes
print amr.items()
