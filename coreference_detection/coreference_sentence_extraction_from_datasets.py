import os
from preprocessing import SentenceAMRPairsExtractor
from AMRGraph import AMR
import AMRData

import numpy as np
import matplotlib.pyplot as plt

import pickle

import pprint


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


def generate_barchart(datasets_coref_statistics_map):
    # map: key = dataset name - value = (total nr of examples, nr of examples with coreferences)

    names_of_datasets = tuple(datasets_coref_statistics_map.keys())
    nr_of_total_dataset_examples = []
    nr_of_dataset_examples_with_coreferences = []

    for dataset_name in names_of_datasets:
        nr_of_total_dataset_examples.append(datasets_coref_statistics_map.get(dataset_name)[0])
        nr_of_dataset_examples_with_coreferences.append(datasets_coref_statistics_map.get(dataset_name)[1])

    nr_of_total_dataset_examples = tuple(nr_of_total_dataset_examples)
    nr_of_dataset_examples_with_coreferences = tuple(nr_of_dataset_examples_with_coreferences)

    # y_indexes = np.arange(0, len(nr_of_total_dataset_examples) * 2, 2)  # the x locations for the groups
    y_indexes = np.arange(len(nr_of_total_dataset_examples))  # the x locations for the groups

    bar_width = 0.45  # the bar_width of the bars

    fig, ax = plt.subplots()

    rectangles1 = ax.bar(y_indexes - bar_width / 2, nr_of_total_dataset_examples, bar_width,
                         color='SkyBlue', label='Total nr. of records')
    rectangles2 = ax.bar(y_indexes + bar_width / 2, nr_of_dataset_examples_with_coreferences, bar_width,
                         color='IndianRed', label='Nr. of records with co-references')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Nr. of dataset records')
    ax.set_xlabel('Dataset names')
    ax.set_title('Co-reference statistics for dataset records')
    ax.set_xticks(y_indexes)
    ax.set_xticklabels(names_of_datasets)
    ax.legend()

    autolabel(rectangles1, ax)
    autolabel(rectangles2, ax)

    plt.show()
    # fig.savefig("dataset_co-references_plot.png")


def get_dataset_names_ordered_the_same_as_dataset_file_names(dataset_file_names, dataset_short_names):
    ordered_dataset_names = []

    for dataset_fname in dataset_file_names:
        for dataset_name in dataset_short_names:
            if dataset_name in dataset_fname:
                ordered_dataset_names.append(dataset_name)
                break

    return ordered_dataset_names


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


def get_merged_amr_coreferenced_nodes_and_sentence_to_write_in_file(amr_str, sentence, coref_nodes):
    file_text = "# sentence: " + sentence + '\n'
    file_text += "# co-referenced nodes: " + " ".join(coref_nodes) + '\n'
    file_text += amr_str + '\n\n'

    return file_text


def generate_files_with_sentences_which_have_coreferences(dataset_names, datasets_directory_path,
                                                          write_corefs_to_files=False, generate_barchart=False):
    '''
    This function generates statistics about the co-referenced records in the datasets and if 'generata_barchart'
    variable is True it prints a barchart graph, which shows the total number of records which have co-references
    in a dataset

    :param dataset_names: string of dataset names abbreviation
    :param datasets_directory_path: path to the directory where the datasets are located
    :param write_corefs_to_files: boolean: if True this function will generate 2 files:
            - 1 file will have all the sentences which have co-references
            - 1 file will have the AMRs which have co-references together with the sentences and the co-referenced nodes
            - 1 binary file which will dump the following strcture with pickle:
                a dictionary which will have the dataset name as key and its value will be a
                a list of tuples as follows: (sentence, nr_of_amr_coreferenced_nodes)
                - e.g. 'bolt':[('My sister walks her dog', 1), ...]
                 - only 'sister' and 'her' are co-references and they point to the same node
    :param generate_barchart: boolean: if True it will generate a barchart graph which has the total number of records
            from a dataset agains the total number of records which have co-references
    :return: void
    '''

    if not os.path.isdir(datasets_directory_path):
        print("Given dataset directory path is not valid!")
        return

    coreference_sentences_dir_path = "./sentences_with_coreferences"

    if not os.path.exists(coreference_sentences_dir_path):
        os.mkdir(coreference_sentences_dir_path, 0o776)

    directory_content = os.listdir(datasets_directory_path)

    # dataset files are .txt files and they contain the dataset name in their file name
    dataset_file_names = filter(
        lambda x: x.endswith('.txt') and any(substring in x for substring in dataset_names), directory_content)

    ordered_dataset_names = get_dataset_names_ordered_the_same_as_dataset_file_names(dataset_file_names, dataset_names)

    # key = dataset name - value = (total nr of examples, nr of examples with coreferences)
    coref_datasets_statistics_map = {}

    # key = dataset name, e.g. 'bolt' - value = list of tuples (sentence, nr_of_coreferenced_nodes)
    sentences_concepts_coref_dict = {}

    # pretty_printer = pprint.PrettyPrinter(indent=4)

    for dataset_file_name, dataset_name in zip(dataset_file_names, ordered_dataset_names):
        dataset_file_path = os.path.join(datasets_directory_path, dataset_file_name)

        sentence_amr_triples = SentenceAMRPairsExtractor.extract_sentence_amr_pairs(dataset_file_path)

        sentences_with_coreferences = []
        amrs_with_coreferences = []

        for sentence, amr_str, amr_id in sentence_amr_triples:
            try:
                amr = AMR.parse_string(amr_str)

                custom_amr = AMRData.CustomizedAMR()
                custom_amr.create_custom_AMR(amr)

                coreferenced_nodes = get_coreferenced_nodes(custom_amr.relations_dict)

                if len(coreferenced_nodes) > 0:
                    # some nodes appear more than 1 time -> those nodes are co-referenced
                    sentences_with_coreferences.append(sentence + '\n\n')

                    #pretty_printer.pprint(custom_amr.relations_dict)

                    if write_corefs_to_files:
                        if sentences_concepts_coref_dict.get(dataset_name):
                            sentences_concepts_coref_dict[dataset_name].append((sentence, len(coreferenced_nodes)))
                        else:
                            sentences_concepts_coref_dict[dataset_name] = [(sentence, len(coreferenced_nodes))]

                        amrs_with_coreferences.append(
                            get_merged_amr_coreferenced_nodes_and_sentence_to_write_in_file(amr_str,
                                                                                            sentence,
                                                                                            coreferenced_nodes)
                        )

            except Exception as e:
                #print(e.message)
                pass  # don't take this sentence into consideration

        coref_datasets_statistics_map[dataset_name] = (len(sentence_amr_triples), len(sentences_with_coreferences))

        if len(sentences_with_coreferences) > 0 and write_corefs_to_files:
            coref_sentences_path = os.path.join(coreference_sentences_dir_path, dataset_file_name + ".coref")
            coref_amrs_path = os.path.join(coreference_sentences_dir_path, dataset_file_name + ".amr.coref")

            # write the sentences which were found to have co-references to file
            with open(coref_sentences_path, "w") as f:
                f.writelines(sentences_with_coreferences)

            # write the corresponding AMRs to file
            with open(coref_amrs_path, "w") as f:
                f.writelines(amrs_with_coreferences)

    if write_corefs_to_files:
        coref_sentences_nodes_dump_path = "coreferenced_sentences_and_nodes.dump"

        with open(coref_sentences_nodes_dump_path, "wb") as f:
            pickle.dump(sentences_concepts_coref_dict, f)

    # # codul de mai jos l-am folosit pt generarea barchart-ului: mi-am salvat nr-ul total de propozitii si nr-ul de
    # # propozitii care contin co-referinte ca sa nu trebuiasca sa parcurg toate dataset-urile de
    # # fiecare data cand vreau sa ajustez codul pt. generarea graficului - 'generate_barchart'
    # with open("coreference_datasets_statistics_bin.dump", "wb") as f:
    #     pickle.dump(coref_datasets_statistics_map, f)

    # with open("coreference_datasets_statistics_bin.dump", "rb") as f:
    #     ds = pickle.load(f)
    #
    # generate_barchart(ds)

    if generate_barchart:
        generate_barchart(coref_datasets_statistics_map)


if __name__ == "__main__":
    dataset_names = ['xinhua', 'bolt', 'proxy', 'dfa', 'cctv', 'guidelines', 'mt09sdl', 'wb', 'consensus']
    datasets_directory_path = '../resources/alignments/unsplit/'

    generate_files_with_sentences_which_have_coreferences(dataset_names, datasets_directory_path, write_corefs_to_files=True)

