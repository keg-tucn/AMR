import re
import copy

from models.amr_graph import AMR


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

    for key, value in list(amr_str.node_to_concepts.items()):
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
        # make sure it doesn"t take into consideration one letter conceps (following "/ ")
        slash_pos = amr_str_lines[k].find("/ ")
        if slash_pos == -1:
            amr_str_lines[k] = re.sub((" " + var_final[k]) + "\n",
                                      " (COREF_" + var_final[k] + " / " + amr_str.node_to_concepts[
                                          var_final[k]] + ")" + " \n", amr_str_lines[k])

    for k in range(0, len(amr_str_lines)):
        slash_pos = amr_str_lines[k].find("/ ")
        if slash_pos == -1:
            amr_str_lines[k] = re.sub((" " + var_final[k]) + "[ ]{1,}\n",
                                      " (COREF_" + var_final[k] + " / " + amr_str.node_to_concepts[
                                          var_final[k]] + ")" + "\n", amr_str_lines[k])

    # coreferinta - > variable)
    for k in range(0, len(amr_str_lines)):
        slash_pos = amr_str_lines[k].find("/ ")
        if slash_pos == -1:
            amr_str_lines[k] = re.sub((" " + var_final[k]) + "\)",
                                      " (COREF_" + var_final[k] + " / " + amr_str.node_to_concepts[
                                          var_final[k]] + ")" + ")", amr_str_lines[k])
    for k in range(0, len(amr_str_lines)):
        slash_pos = amr_str_lines[k].find("/ ")
        if slash_pos == -1:
            amr_str_lines[k] = re.sub((" " + var_final[k]) + "~",
                                      " (COREF_" + var_final[k] + " / " + amr_str.node_to_concepts[
                                          var_final[k]] + ")" + "~", amr_str_lines[k])

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
        elif not line.startswith("#"):
            cur_amr.append(line.strip())

    if cur_amr:  # file did not end with newline, so add AMR here
        all_amrs.append(" ".join(cur_amr).strip())

    return all_amrs
