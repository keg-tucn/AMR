from models import AMRData
from postprocessing import ActionSequenceReconstruction
from preprocessing import ActionSequenceGenerator
import re
from models.AMRGraph import AMR
import copy
from smatch import smatch_util, smatch_amr


def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


def collectSentences(path_input_file):
    with open(path_input_file) as input_file:
        for line in input_file:
            if line.__contains__("# ::tok"):
                new_line = line.replace("# ::tok", "")
                print(new_line)
            elif line.__contains__("# ::snt"):
                new_line = line.replace("# ::snt", "")
                print(new_line)


def baseline(amr):
    amr_copy = copy.deepcopy(amr)
    amr_str = AMR.parse_string(amr_copy)
    amr_str_lines = amr_copy.splitlines()

    variables = list()
    variables_compare = list()
    var_final = list()

    for key, value in amr_str.node_to_concepts.iteritems():
        variable = key
        variables.append(variable)

    seqIDList = amr_str.get_seqID()
    for k in range(0, len(seqIDList)):
        variable_compare = amr_str.get_variable(seqIDList[k])
        variables_compare.append(variable_compare)

    for k in range(0, len(variables_compare)):
        for j in range(0, len(variables)):
            if variables_compare[k] == (variables[j]):
                variable_final = variables_compare[k]
                var_final.append(variable_final)

    # coreferinta - > variable + newLine
    for k in range(0, len(amr_str_lines)):
        # make sure it doesn't take into consideration one letter conceps (following "/ ")
        slash_pos = amr_str_lines[k].find("/ ")
        if slash_pos == -1:
            amr_str_lines[k] = re.sub((" " + var_final[k]) + '\n',
                                      " (COREF_" + var_final[k] + " / " + amr_str.node_to_concepts[
                                          var_final[k]] + ")" + ' \n', amr_str_lines[k])

    for k in range(0, len(amr_str_lines)):
        slash_pos = amr_str_lines[k].find("/ ")
        if slash_pos == -1:
            amr_str_lines[k] = re.sub((" " + var_final[k]) + '[ ]{1,}\n',
                                      " (COREF_" + var_final[k] + " / " + amr_str.node_to_concepts[
                                          var_final[k]] + ")" + '\n', amr_str_lines[k])

    # coreferinta - > variable)
    for k in range(0, len(amr_str_lines)):
        slash_pos = amr_str_lines[k].find("/ ")
        if slash_pos == -1:
            amr_str_lines[k] = re.sub((" " + var_final[k]) + '\)',
                                      " (COREF_" + var_final[k] + " / " + amr_str.node_to_concepts[
                                          var_final[k]] + ")" + ')', amr_str_lines[k])
    for k in range(0, len(amr_str_lines)):
        slash_pos = amr_str_lines[k].find("/ ")
        if slash_pos == -1:
            amr_str_lines[k] = re.sub((" " + var_final[k]) + '\~',
                                      " (COREF_" + var_final[k] + " / " + amr_str.node_to_concepts[
                                          var_final[k]] + ")" + '~', amr_str_lines[k])

    new_amr_str = " "
    for k in range(0, len(amr_str_lines)):
        new_amr_str += amr_str_lines[k] + "\n"
    new_amr_str = copy.deepcopy(new_amr_str)

    return new_amr_str


def reentrancy_restoring(amr_str):
    new_amr_lines = amr_str.splitlines()

    coref_string = "COREF_"
    for k in range(0, len(new_amr_lines)):
        # print(new_amr_lines[k])
        if new_amr_lines[k].__contains__(coref_string):
            old_substr = "(" + find_between(new_amr_lines[k], "(", ")") + ")"
            variable = find_between(new_amr_lines[k], coref_string, " ")

            new_amr_lines[k] = new_amr_lines[k].replace(old_substr, variable)

    new_amr_str = " "
    for k in range(0, len(new_amr_lines)):
        new_amr_str += new_amr_lines[k] + "\n"
    new_amr_str = copy.deepcopy(new_amr_str)

    return new_amr_str


def del_sents_tabs_newlines(f):
    all_amrs = []
    cur_amr = []

    for line in open(f, "r"):
        if not line.strip() and cur_amr:
            cur_amr_line = " ".join(cur_amr)
            all_amrs.append(cur_amr_line.strip())
            cur_amr = []
        elif not line.startswith('#'):
            cur_amr.append(line.strip())

    if cur_amr:  # file did not end with newline, so add AMR here
        all_amrs.append(" ".join(cur_amr).strip())

    return all_amrs


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

    new_amr_parsed = AMR.parse_string(new_amr_str)
    custom_AMR = AMRData.CustomizedAMR()
    custom_AMR.create_custom_AMR(new_amr_parsed)

    print(custom_AMR.tokens_to_concepts_dict)
    print(custom_AMR.relations_dict)
    print(new_amr_parsed.node_to_concepts)
    print(new_amr_parsed.node_to_tokens)

    action_sequence = ActionSequenceGenerator.generate_action_sequence(custom_AMR, sentence)
    print action_sequence

    generated_amr_str = ActionSequenceReconstruction.reconstruct_all(action_sequence)

    after_post_amr = reentrancy_restoring(generated_amr_str)

    smatch_results = smatch_util.SmatchAccumulator()
    original_amr = smatch_amr.AMR.parse_AMR_line(amr)
    generated_amr = smatch_amr.AMR.parse_AMR_line(after_post_amr)
    print "generated amr"
    print generated_amr
    smatch_f_score = smatch_results.compute_and_add(generated_amr, original_amr)
    print "smatch: "
    print smatch_f_score
