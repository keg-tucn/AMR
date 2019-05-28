from Baseline import baseline, reentrancy_restoring
from models import amr_graph, amr_data
from preprocessing import ActionSequenceGenerator
from postprocessing import action_sequence_reconstruction
from smatch import smatch_util, smatch_amr

if __name__ == "__main__":
    path_file = "delVars.txt"
    sentence = "I remain poised"
    amr = """(r / remain-01~e.1 
      :ARG1 (i / i~e.0) 
      :ARG2 (p / poise-01~e.2 
            :ARG1 i~e.0))"""

    new_amr_str = baseline(amr)

    print("before baseline:")
    print(amr)
    print("\n")

    print("after baseline:")
    print(new_amr_str)
    print("\n")

    # after_post_amr = reentrancy_restoring(new_amr, , variables, n_to_concepts, n_to_tokens,  rel_to_tokens,info_for_coref)

    new_amr_parsed = amr_graph.AMR.parse_string(new_amr_str)
    custom_AMR = amr_data.CustomizedAMR()
    custom_AMR.create_custom_AMR(new_amr_parsed)

    print(custom_AMR.tokens_to_concepts_dict)
    print(custom_AMR.relations_dict)
    print(new_amr_parsed.node_to_concepts)
    print(new_amr_parsed.node_to_tokens)

    action_sequence = ActionSequenceGenerator.generate_action_sequence(custom_AMR, sentence)
    print action_sequence

    generated_amr_str = action_sequence_reconstruction.reconstruct_all(action_sequence)

    after_post_amr = reentrancy_restoring(generated_amr_str)

    smatch_results = smatch_util.SmatchAccumulator()
    original_amr = smatch_amr.AMR.parse_AMR_line(amr)
    generated_amr = smatch_amr.AMR.parse_AMR_line(after_post_amr)
    print "generated amr"
    print generated_amr
    smatch_f_score = smatch_results.compute_and_add(generated_amr, original_amr)
    print "smatch: "
    print smatch_f_score
