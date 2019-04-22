from os import path
import pickle as js

from definitions import CONCEPTS_RELATIONS_DICT
from feature_extraction import dataset_loader


def get_concepts_relations_pairs():
    if path.exists(path.dirname(CONCEPTS_RELATIONS_DICT)):
        with open(CONCEPTS_RELATIONS_DICT, "rb") as dict_file:
            return js.load(dict_file)
    else:
        return extract_concepts_relations_pairs()


def extract_concepts_relations_pairs():
    """
    Return a dictionary mapping (parent_concept, child_concept) -> parent_child_AMR_relation
    after parsing the entire dataset and also save the dictionary in the resources folder
    """
    dataset_identifier = "r2"
    dataset_splits = ["training", "dev", "test"]

    concepts_relations_dict = dict()

    for data_split in dataset_splits:
        graphs_data = dataset_loader.read_original_graphs(type=data_split, filter_path=dataset_identifier)

        for graph_data in graphs_data:
            amr_graph = graph_data[2]

            for concept_relations in amr_graph.items():
                for concept_relation in concept_relations[1].items():
                    parent_token = concept_relations[0]
                    child_token = concept_relation[1][0]

                    parent_concept = amr_graph.node_to_concepts[parent_token]
                    if child_token in amr_graph.node_to_concepts:
                        child_concept = amr_graph.node_to_concepts[child_token]
                    else:
                        child_concept = child_token

                    if (parent_concept, child_concept) in concept_relations:
                        concepts_relations_dict[(parent_concept, child_concept)].append(concept_relation[0])
                    else:
                        concepts_relations_dict[(parent_concept, child_concept)] = [concept_relation[0]]

    with open(CONCEPTS_RELATIONS_DICT, "wb") as dict_file:
        js.dump(concepts_relations_dict, dict_file)

    return concepts_relations_dict


if __name__ == "__main__":

    concepts_rels_dict = extract_concepts_relations_pairs()

    multiple_rels_pairs = 0

    for item in concepts_rels_dict.items():
        if len(item[1]) > 1:
            print str(item[0]) + " | " + str(item[1])
            multiple_rels_pairs += 1

    print "%d / %d" % (multiple_rels_pairs, len(concepts_rels_dict.items()))
