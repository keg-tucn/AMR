import spacy
import neuralcoref

import numpy as np
import matplotlib.pyplot as plt

import pickle
import copy
import re

from AMRGraph import AMR
import AMRData

coref_sentences_nodes_dump_path = "coreferenced_sentences_and_nodes.dump"
neural_coref_statistics_dump_path = "neural_coref_statistics.dump"
coref_plot_path = "neural_coref_dataset_co-references_plot.png"


def autolabel(rects, ax):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.

    https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """

    xpos = 'center'
    offset = {'center': 0.5}

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
                '{}'.format(height), ha=xpos, va='bottom')


def generate_barchart(neural_coref_statistics_map):
    # dictionary which has as key the dataset name and as value a tuple of the form: (a,b,c), where:
    # a = nr of sentences which have co-references
    # b = nr of sentences in which neuralCoref found co-references
    # c = nr of sentences in which neuralCoref found co-references and the number of 'coref_clusters' found
    #     is equal to the number of co-referenced amr nodes

    names_of_datasets = ['xinhua', 'bolt', 'proxy', 'guidelines', 'dfa', 'cctv', 'mt09sdl', 'wb', 'consensus']
    #names_of_datasets = tuple(neural_coref_statistics_map.keys())
    nr_of_inital_examples_with_corefs = []
    nr_of_examples_with_coreferences_found_by_neural_coref = []
    # nr_of_examples_with_the_same_nr_of_coref_clusters_as_coreferenced_nodes = []

    for dataset_name in names_of_datasets:
        nr_of_inital_examples_with_corefs.append(neural_coref_statistics_map.get(dataset_name)[0])
        nr_of_examples_with_coreferences_found_by_neural_coref.append(neural_coref_statistics_map.get(dataset_name)[1])
        # nr_of_examples_with_the_same_nr_of_coref_clusters_as_coreferenced_nodes.append(
        #     neural_coref_statistics_map.get(dataset_name)[2]
        # )

    nr_of_inital_examples_with_corefs = tuple(nr_of_inital_examples_with_corefs)
    nr_of_examples_with_coreferences_found_by_neural_coref = tuple(nr_of_examples_with_coreferences_found_by_neural_coref)
    # nr_of_examples_with_the_same_nr_of_coref_clusters_as_coreferenced_nodes = tuple(nr_of_examples_with_the_same_nr_of_coref_clusters_as_coreferenced_nodes)

    # y_indexes = np.arange(0, len(nr_of_total_dataset_examples) * 2, 2)  # the x locations for the groups
    y_indexes = np.arange(len(nr_of_inital_examples_with_corefs))  # the x locations for the groups

    bar_width = 0.45  # the bar_width of the bars

    fig, ax = plt.subplots()

    rectangles1 = ax.bar(y_indexes - bar_width / 2, nr_of_inital_examples_with_corefs, bar_width,
                         color='SkyBlue', label='Initial nr. of records\nwhich have co-references')
    rectangles2 = ax.bar(y_indexes + bar_width / 2, nr_of_examples_with_coreferences_found_by_neural_coref, bar_width,
                         color='IndianRed', label='Nr. of records,\nwhich have co-references,\nfound by neuralCoref')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Nr. of dataset records')
    ax.set_xlabel('Dataset names')
    ax.set_title('Co-reference statistics for dataset records\n-with neuralCoref-')
    ax.set_xticks(y_indexes)
    ax.set_xticklabels(names_of_datasets)
    ax.legend()

    autolabel(rectangles1, ax)
    autolabel(rectangles2, ax)

    plt.show()
    fig.savefig(coref_plot_path)

def find_coreferences_in_input_data(save_results = False):
    # Load the largest English SpaCy model
    nlp = spacy.load('en_core_web_lg')

    # Add neural coref to SpaCy's pipe
    neuralcoref.add_to_pipe(nlp, blacklist=False)


    with open(coref_sentences_nodes_dump_path, "rb") as f:
        # key = dataset name, e.g. 'bolt' - value = list of tuples (sentence, nr_of_coreferenced_nodes)
        coref_sentences_concepts_dict = pickle.load(f)

    # key is the same as in the 'coref_sentences_concepts_dict' and the value is a list of spaCy Doc objects
    doc_objects_dict = {}

    # dictionary which has as key the dataset name and as value a tuple of the form: (a,b,c), where:
    # a = nr of sentences which have co-references
    # b = nr of sentences in which neuralCoref found co-references
    # c = nr of sentences in which neuralCoref found co-references and the number of 'coref_clusters' found
    #     is equal to the number of co-referenced amr nodes
    coref_statistics = {}

    for key_dataset_name in coref_sentences_concepts_dict.keys():
        nr_of_sentences_in_which_corefs_were_found = 0
        nr_of_sentences_which_have_same_nr_of_coref_clusters_as_coreferenced_amr_nodes = 0

        for sentence, nr_of_coreferenced_amr_nodes in coref_sentences_concepts_dict[key_dataset_name]:
            doc = nlp(unicode(sentence, 'utf-8'))

            if doc_objects_dict.get(key_dataset_name):
                doc_objects_dict[key_dataset_name].append(doc)
            else:
                doc_objects_dict[key_dataset_name] = [doc]

            if doc._.has_coref:
                nr_of_sentences_in_which_corefs_were_found += 1

                if len(doc._.coref_clusters) == nr_of_coreferenced_amr_nodes:
                    nr_of_sentences_which_have_same_nr_of_coref_clusters_as_coreferenced_amr_nodes += 1

        coref_statistics[key_dataset_name] = (
            len(coref_sentences_concepts_dict[key_dataset_name]),
            nr_of_sentences_in_which_corefs_were_found,
            nr_of_sentences_which_have_same_nr_of_coref_clusters_as_coreferenced_amr_nodes
        )

    if save_results:
        with open(neural_coref_statistics_dump_path, "wb") as f:
            pickle.dump(coref_statistics, f)


def get_coreferenced_nodes(custom_amr_relations_dict):
    coreferenced_nodes = {}
    nodes = {}

    # key is a (node,parent) tuple -> licenta silviana pag. 61
    # we don't need the parent
    for node, _ in custom_amr_relations_dict.keys():
        if nodes.get(node) and len(node) <= 3:  # we want to get only nodes, i.e. 'a', 'c', 'c2'
            # we just need the node -> this is why we use None...
            # in case the node is referenced more than once we will not add it again (it will be overwritten)
            coreferenced_nodes[node] = None
        else:
            nodes[node] = True

    return list(coreferenced_nodes.keys())


def has_AMR_coreferenced_nodes(amr_graph):
    nodes = {}  # all nodes of the AMR

    amr = AMR.parse_string(amr_graph)

    custom_amr = AMRData.CustomizedAMR()
    custom_amr.create_custom_AMR(amr)

    for node, _ in custom_amr.relations_dict.keys():
        if nodes.get(node) and len(node) <= 3:  # we want to get only nodes, i.e. 'a', 'c', 'c2'
            # we have found 'node' previously -> 'node' is a co-referenced node
            return True
        else:
            nodes[node] = True

    return False


def get_all_amr_variables(amr):
    variables = []

    for variable, _ in amr.node_to_concepts.iteritems():
        variables.append(variable)

    return variables


def get_amr_variables_found_on_each_row_in_the_amr_string(amr, variables):
    # on each row of the AMR there is only one variable corresponding to the concept present on that row
    # variables_on_each_row = [var for var in amr.get_all_variables() if var in variables]
    #
    # for variable in amr.get_variables():
    #     if variable in variables:
    #         variables_on_each_row.append(variable)

    # return variables_on_each_row
    return [var for var in amr.get_all_variables() if var in variables]


def duplicate_concept_data_on_amr_for_coreferenced_nodes(original_amr_str):
    amr_str = copy.deepcopy(original_amr_str)
    amr_str_lines = amr_str.splitlines()

    # get rid of the trailing whitespaces
    amr_str_lines = [line.rstrip() for line in amr_str_lines]

    amr = AMR.parse_string(amr_str)

    variables = get_all_amr_variables(amr)
    variables_on_each_row = get_amr_variables_found_on_each_row_in_the_amr_string(amr, variables)

    # dictionary: key = variables, value = count of co-referenced variable (COREF_var_1, COREF_var_2, ...)
    var_count_dict = dict((variables[i],0) for i in range(0, len(variables) ))

    for i in range(0, len(amr_str_lines)):
        # co-referenced variables don't have a concept, i.e., after the variable there should not be a '/'
        # a '/' character can be found only after one of the co-referenced variables, where the concept is also defined

        # if we don't find a '/', i.e., the current line has a co-referenced variable
        if amr_str_lines[i].find("/ ") == -1:
            # after the variable there can be:  1) one or more closing parentheses ')'
            #                                   2) an alignment, i.e., '~e...'
            #                                   3) nothing (the variable is the last char of the line) -> treat it last

            coref_pattern = ' (COREF_'

            amr_str_lines[i] = re.sub(
                ' ' + variables_on_each_row[i] + '\)',
                coref_pattern + variables_on_each_row[i] + '_' + str(var_count_dict[variables_on_each_row[i]]) + ' / ' +
                amr.node_to_concepts[variables_on_each_row[i]] + '))',  # put back the replaced ')'
                amr_str_lines[i])

            # if we have an alignment (~e..), we can't put our closing parenthesis right before the '~', but
            # we need to put it after the alignment -> we will insert it before the last char, which is '\n'
            str_substitution_with_alignment = re.sub(
                ' ' + variables_on_each_row[i] + '~',
                coref_pattern + variables_on_each_row[i] + '_' + str(var_count_dict[variables_on_each_row[i]]) + ' / ' +
                amr.node_to_concepts[variables_on_each_row[i]] + '~',  # put back the replaced '~'
                amr_str_lines[i])
            
            if str_substitution_with_alignment != amr_str_lines[i]:
                # a substitution when we have an alignment was done -> insert the closing parenthesis at the end
                last_char_idx = len(str_substitution_with_alignment)

                # convert string to list of chars to be able to insert ')'
                str_substitution_with_alignment = list(str_substitution_with_alignment)

                str_substitution_with_alignment.insert(last_char_idx, ')')

                # assign the new string to the line
                amr_str_lines[i] = ''.join(str_substitution_with_alignment)

            amr_str_lines[i] = re.sub(
                ' ' + variables_on_each_row[i] + '$', # $ = end of line
                coref_pattern + variables_on_each_row[i] + '_' + str(var_count_dict[variables_on_each_row[i]]) + ' / ' +
                amr.node_to_concepts[variables_on_each_row[i]] + ')',
                amr_str_lines[i])

            # increment the COREF count for the co-referenced variable
            var_count_dict[variables_on_each_row[i]] += 1

    return "\n".join(amr_str_lines)


# we should also have the AMR.parse_string dict to change where we have made modifications....
def restore_coreferenced_nodes_in_amr(predicted_amr_str):
    coref_pattern_start = '(COREF_'

    amr_str = copy.deepcopy(predicted_amr_str)

    if amr_str.find(coref_pattern_start) == -1:
        return predicted_amr_str

    amr_str_lines = amr_str.splitlines()

    # get rid of the trailing whitespaces
    amr_str_lines = [line.rstrip() for line in amr_str_lines]

    for i, amr_line in enumerate(amr_str_lines):
        pattern_start_idx = amr_line.find(coref_pattern_start)

        if pattern_start_idx != -1:
            # the variable is stored between 2 '_' chars, i.e., COREF_a_0
            variable = amr_line.split('_')[1]

            # now we can have 2 cases:
            #   either the concept is followed by a '~' char and an allignment
            #   either the concept is followed by a ')' char and maybe more ')' chars after

            # find will start searching for char from 'pattern_start_idx'
            tilda_char_idx = amr_line.find('~', pattern_start_idx)
            parenthesis_char_idx = amr_line.find(')', pattern_start_idx)

            if tilda_char_idx != -1:
                # get everything from '~' onwards and delete the trailing ')'
                restored_line = amr_line[:pattern_start_idx] + variable + amr_line[tilda_char_idx:-1]
            elif parenthesis_char_idx != -1:
                restored_line = amr_line[:pattern_start_idx] + variable + amr_line[parenthesis_char_idx + 1:]
            else:
                raise Exception('AMR String is corrupted!')

            amr_str_lines[i] = restored_line

    return "\n".join(amr_str_lines)


def remove_wiki_and_name_concepts_from_amr(original_amr_str):
    amr_str = copy.deepcopy(original_amr_str)
    amr_str_lines = amr_str.splitlines()

    # get rid of the trailing whitespaces
    amr_str_lines = [line.rstrip() for line in amr_str_lines]

    i = 0
    replaced_wiki = False
    replaced_name = False

    while i < len(amr_str_lines):
        # new_line = amr_str_lines[i]

        pos_found_wiki = amr_str_lines[i].find(" :wiki")  # keep the space before :wiki because at that position we will
                                                          # add the trailing closing parentheses of :name

        if pos_found_wiki != -1:
            # check to see if :name is on the same row
            pos_found_name = amr_str_lines[i].find(":name")

            if pos_found_name != -1:
                if pos_found_name < pos_found_wiki:
                    # name should be after wiki -> ERROR -> terminate this operation
                    print(':name occurred before :wiki at line {} -> ERROR'.format(i))

                    return original_amr_str

                pos_name_closing_parenthesis = amr_str_lines[i].find(')', pos_found_name)

                if pos_name_closing_parenthesis == -1:
                    print(":name does NOT have a ') -> ERROR'")
                    return original_amr_str

                # replace both :wiki and :name
                if pos_name_closing_parenthesis < len(amr_str_lines[i]) - 1:
                    new_line = amr_str_lines[i][:pos_found_wiki] + amr_str_lines[i][pos_name_closing_parenthesis + 1:]
                else:
                    # there isn't anything after ')' of :name so we remove everything from :wiki onwards
                    new_line = amr_str_lines[i][:pos_found_wiki]

                replaced_wiki = True
                replaced_name = True

            if not replaced_wiki:
                new_line = amr_str_lines[i][:pos_found_wiki]
                replaced_wiki = True

            if not replaced_name:
                # check the next row to replace :name and to add closing parenthesis if any (except the one from :name)
                # to the current row on which :wiki was found
                if i + 1 == len(amr_str_lines):
                    print(':name is not on the row below :wiki (no more rows below :wiki) -> ERROR')
                    return original_amr_str

                # remove from the beginning and end of the string whitespaces and tabs
                amr_str_row_below_wiki_stripped = amr_str_lines[i + 1].strip(' \t')

                pos_found_name = amr_str_row_below_wiki_stripped.find(":name")

                if pos_found_name == -1:
                    print(':name is not on the row below :wiki -> ERROR')
                    return original_amr_str

                if pos_found_name != 0:
                    # we check this to be certain that only the concept :name is on this row
                    print(':name is not at the beginning of the stripped row -> ERROR')
                    return original_amr_str

                pos_name_closing_parenthesis = amr_str_row_below_wiki_stripped.find(')', pos_found_name)

                if pos_name_closing_parenthesis == -1:
                    print(":name does NOT have a ') -> ERROR'")
                    return original_amr_str

                # replace both :name and delete row which had name, because that row
                if pos_name_closing_parenthesis < len(amr_str_row_below_wiki_stripped) - 1:
                    new_line += amr_str_row_below_wiki_stripped[pos_name_closing_parenthesis + 1:]
                # else:
                    # there isn't anything after ')' of :name so we do nothing, since this line will be deleted

                del amr_str_lines[i + 1]

            amr_str_lines[i] = new_line

            replaced_name = False
            replaced_wiki = False

        i += 1

    return "\n".join(amr_str_lines)


if __name__ == "__main__":

    amr_string = \
    """(a / and~e.16 
      :op1 (e / entrust-01~e.5 
            :ARG1 (p / project~e.3 
                  :mod (e2 / export-01~e.1) 
                  :mod (c / credit~e.2) 
                  :mod (t / this~e.0)) 
            :ARG2~e.6 (c4 / company~e.13 :wiki "Bank_of_China" 
                  :name (n / name :op1 "Bank"~e.8 :op2 "of"~e.9 :op3 "China"~e.10) 
                  :ARG0-of~e.13 (a2 / act-01~e.13 
                        :ARG1 (b / bank~e.14 
                              :ARG0-of (a4 / act-02~e.13))))) 
      :op2 (a3 / agent~e.19 
            :mod (c3 / commerce~e.18) 
            :poss~e.17 p~e.17 
            :domain~e.20 (c2 / company :wiki - 
                  :name (n2 / name :op1 "China"~e.22 :op2 "Electronics"~e.23 :op3 "Import"~e.24 :op4 "and"~e.25 :op5 "Export"~e.26 :op6 "Corporation"~e.27))))"""

    amr_string2 = \
    '''(a / and 
      :op1 (v / value-01~e.5 
            :ARG1~e.6 (t / thing~e.9 
                  :ARG1-of~e.9 (e / export-01~e.9 
                        :ARG0~e.8 (c / country :wiki "Taiwan" 
                              :name (n / name :op1 "Taiwan"~e.7)) 
                        :ARG2~e.10 (m / mainland~e.12))) 
            :ARG2 (m2 / monetary-quantity :quant 17800000000 
                  :unit (d / dollar~e.36 
                        :mod (c4 / country :wiki "United_States" 
                              :name (n2 / name :op1 "US"~e.35)))) 
            :ARG1-of (i / increase-01~e.20 
                  :ARG2~e.21 (p / percentage-entity~e.23 :value 20~e.22) 
                  :ARG4 m2~e.14,15,16,17 
                  :ARG1-of (c2 / compare-01~e.24 
                        :ARG2~e.25 (y / year~e.27 
                              :mod (l / last~e.26))))) 
      :op2 (v3 / value-01~e.31 
            :ARG1 (t4 / thing~e.30 
                  :ARG1-of~e.30 (i3 / import-01~e.30)) 
            :ARG2 (m3 / monetary-quantity :quant 3100000000~e.33,34 
                  :unit (d3 / dollar~e.36 
                        :mod c4)) 
            :ARG1-of (i2 / increase-01~e.39 
                  :ARG2~e.40 (p2 / percentage-entity~e.42 :value 74~e.41) 
                  :ARG4 m3 
                  :ARG1-of (c3 / compare-01~e.43 
                        :ARG2~e.44 y~e.45,46))) 
      :part-of (v2 / volume~e.2 
            :mod (t3 / this~e.1)))'''

    amr_after_coref_handl = \
    '''(a / and
      :op1 (v / value-01~e.5
            :ARG1~e.6 (t / thing~e.9
                  :ARG1-of~e.9 (e / export-01~e.9
                        :ARG0~e.8 (c / country :wiki "Taiwan"
                              :name (n / name :op1 "Taiwan"~e.7))
                        :ARG2~e.10 (m / mainland~e.12)))
            :ARG2 (m2 / monetary-quantity :quant 17800000000
                  :unit (d / dollar~e.36
                        :mod (c4 / country :wiki "United_States"
                              :name (n2 / name :op1 "US"~e.35))))
            :ARG1-of (i / increase-01~e.20
                  :ARG2~e.21 (p / percentage-entity~e.23 :value 20~e.22)
                  :ARG4 (COREF_m2_0 / monetary-quantity~e.14,15,16,17)
                  :ARG1-of (c2 / compare-01~e.24
                        :ARG2~e.25 (y / year~e.27
                              :mod (l / last~e.26)))))
      :op2 (v3 / value-01~e.31
            :ARG1 (t4 / thing~e.30
                  :ARG1-of~e.30 (i3 / import-01~e.30))
            :ARG2 (m3 / monetary-quantity :quant 3100000000~e.33,34
                  :unit (d3 / dollar~e.36
                        :mod (COREF_c4_0 / country)))
            :ARG1-of (i2 / increase-01~e.39
                  :ARG2~e.40 (p2 / percentage-entity~e.42 :value 74~e.41)
                  :ARG4 (COREF_m3_0 / monetary-quantity)
                  :ARG1-of (c3 / compare-01~e.43
                        :ARG2~e.44 (COREF_y_0 / year~e.45,46))))
      :part-of (v2 / volume~e.2
            :mod (t3 / this~e.1)))'''

    amr_string3 = \
    '''(i / introduce-02~e.10 
      :ARG0 (s / system~e.3 
            :mod (m / manage-01~e.2 
                  :ARG1 (p / project~e.1)) 
            :poss (p6 / project~e.1,8 :wiki "Three_Gorges_Dam" 
                  :name (n / name :op1 "The"~e.5 :op2 "Three"~e.6 :op3 "Gorges"~e.7))) 
      :ARG1 (a / and~e.14 
            :op1 (m2 / method~e.13 
                  :mod (m3 / manage-01~e.12) 
                  :mod (s2 / science~e.11)) 
            :op2 (t / technology~e.17 
                  :mod (c / computer~e.16) 
                  :ARG1-of (a2 / advanced-02~e.15) 
                  :poss~e.18 (c3 / company~e.26 :wiki - 
                        :name (n3 / name :op1 "MAI"~e.20 :op2 "Corp."~e.21 :op3 "of"~e.22 :op4 "Canada"~e.23) 
                        :ARG0-of~e.26 (u / use-01~e.26 
                              :ARG1 (s3 / system~e.31 
                                    :mod (m4 / manage-01~e.30 
                                          :ARG1 (d / database~e.29 
                                                :mod (p3 / project~e.28)))) 
                              :ARG2~e.36 (a3 / and~e.39 
                                    :op1 (c6 / control-01~e.38 
                                          :ARG0~e.37 t~e.37 
                                          :ARG1~e.41 (p2 / process-02~e.44 
                                                :ARG1~e.45 (a4 / and~e.61 
                                                      :op1 (d2 / design-01~e.46 
                                                            :ARG1~e.63 (s4 / subproject~e.65 
                                                                  :mod (e2 / each~e.64) 
                                                                  :part-of~e.66 p6~e.67,68,69,70)) 
                                                      :op2 (p5 / plan-01~e.48 
                                                            :ARG1 s4) 
                                                      :op3 (c7 / contract-02~e.50 
                                                            :ARG1 s4) 
                                                      :op4 (f / finance-01~e.52 
                                                            :ARG1 s4) 
                                                      :op5 (a5 / and~e.55 
                                                            :op1 (g / good~e.54) 
                                                            :op2 (m6 / material~e.56) 
                                                            :part-of s4) 
                                                      :op6 (e / equip-01~e.58 
                                                            :part-of s4) 
                                                      :op7 (c8 / construct-01~e.60 
                                                            :part-of s4) 
                                                      :op8 (i2 / install-01~e.62 
                                                            :part-of s4)) 
                                                :mod (w / whole~e.43))) 
                                    :op2 (m5 / manage-01~e.40 
                                          :ARG0 t 
                                          :ARG1 p2)) 
                              :prep-as~e.32 (c4 / component~e.35 
                                    :mod (c5 / core~e.34)))))))'''

    print(duplicate_concept_data_on_amr_for_coreferenced_nodes(amr_string3))
    # amr_restored = restore_coreferenced_nodes_in_amr(duplicate_concept_data_on_amr_for_coreferenced_nodes(amr_string2))
    # # amr_handled = duplicate_concept_data_on_amr_for_coreferenced_nodes(amr_string3)
    # # amr_restored = restore_coreferenced_nodes_in_amr(amr_handled)
    #
    # print amr_restored
    # print amr_handled

    amr_str_wiki = '''(a / and~e.16
      :op1 (e / entrust-01~e.5
            :ARG1 (p / project~e.3
                  :mod (e2 / export-01~e.1)
                  :mod (c / credit~e.2)
                  :mod (t / this~e.0))
            :ARG2~e.6 (c4 / company~e.13 :wiki "Bank_of_China"
                  :name (n / name :op1 "Bank"~e.8 :op2 "of"~e.9 :op3 "China"~e.10)
                  :ARG0-of~e.13 (a2 / act-01~e.13
                        :ARG1 (b / bank~e.14
                              :ARG0-of (a4 / act-02~e.13)))))
      :op2 (a3 / agent~e.19
            :mod (c3 / commerce~e.18)
            :poss~e.17 (COREF_p_0 / project~e.17)
            :domain~e.20 (c2 / company :wiki -
                  :name (n2 / name :op1 "China"~e.22 :op2 "Electronics"~e.23 :op3 "Import"~e.24 :op4 "and"~e.25 :op5 "Export"~e.26 :op6 "Corporation"~e.27))))'''

    amr_str_wiki2 = '''(t / talk-01
      :ARG0 (s / she)
      :ARG2 (h / he)
      :medium (l / language :wiki "French_language" :name (n / name :op1 "French")))'''

    amr_str_wiki3 = '''(s / sign-02~e.7
      :ARG0 (g / government-organization~e.6
            :ARG0-of~e.6 (g3 / govern-01~e.6
                  :ARG1 (c3 / country~e.16 :wiki "Burma"
                        :name (n / name :op1 "Myanmar"~e.3))))
      :ARG1 (a / agreement~e.9
            :topic~e.10 (t / trade-01~e.12
                  :ARG0 (COREF_c3_0 / country)
                  :ARG2 (COREF_c4_0 / country)
                  :location (b / border~e.11)))
      :ARG2 (g2 / government-organization~e.6
            :ARG0-of~e.6 (g4 / govern-01~e.6
                  :ARG1 (c4 / country~e.16 :wiki "Thailand"
                        :name (n2 / name :op1 "Thailand"~e.5))))
      :time (a2 / afternoon~e.1
            :mod (t2 / this~e.0))
      :location (h / here~e.17))'''

    # print(remove_wiki_and_name_concepts_from_amr(amr_str_wiki3))

    # print duplicate_concept_data_on_amr_for_coreferenced_nodes(amr_string2)
    #find_coreferences_in_input_data()

    # with open("neural_coref_statistics_2.dump", "rb") as f:
    #     coref_statistics = pickle.load(f)
    #
    # generate_barchart(coref_statistics)
