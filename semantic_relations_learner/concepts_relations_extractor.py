from os import path
import pickle as js
import numpy as np

from definitions import CONCEPTS_RELATIONS_DICT
from data_extraction import dataset_loader


def get_concepts_relations_pairs():
    if path.exists(path.dirname(CONCEPTS_RELATIONS_DICT)):
        with open(CONCEPTS_RELATIONS_DICT, "rb") as dict_file:
	        try:
	            return js.load(dict_file)
	        except Exception:
	            return extract_concepts_relations_pairs()
    return extract_concepts_relations_pairs()


def extract_concepts_relations_pairs(with_relation_frequency=False):
    """
    Return a dictionary mapping (parent_concept, child_concept) -> parent_child_AMR_relation
    after parsing the entire dataset.
    Save the dictionary in the resources folder.
    """
    dataset_identifier = ""
    dataset_splits = ["training", "dev", "test"]

    concepts_relations_dict = dict()

    # iterate over data set splits
    for data_split in dataset_splits:
        graphs_data = dataset_loader.read_original_graphs(file_type=data_split, filter_path=dataset_identifier)
        # iterate over C-AMR graph structures
        for graph_data in graphs_data:
            amr_graph = graph_data[2]

            # iterate over semantic relations dict ( parent_node -> [(relation, [(child_node)])] )
            for (parent_token, parent_relations) in list(amr_graph.items()):
                # iterate over relations of a node
                for relation in list(parent_relations.items()):
                    child_token = relation[1][0]
                    relation_type = relation[0]

                    parent_concept = amr_graph.node_to_concepts[parent_token]

                    # the child token doesn't correspond to a proper name, which is not mapped in the dictionary
                    if child_token in amr_graph.node_to_concepts:
                        child_concept = amr_graph.node_to_concepts[child_token]

                        if with_relation_frequency:
                            if (parent_concept, child_concept) in concepts_relations_dict:
                                found = False
                                for (rel_type, rel_count) in concepts_relations_dict[(parent_concept, child_concept)]:
                                    if relation_type == rel_type:
                                        concepts_relations_dict[(parent_concept, child_concept)].remove(
                                            (rel_type, rel_count))
                                        concepts_relations_dict[(parent_concept, child_concept)].append(
                                            (rel_type, rel_count + 1))
                                        found = True
                                        break
                                if not found:
                                    concepts_relations_dict[(parent_concept, child_concept)].append((relation_type, 1))
                            else:
                                concepts_relations_dict[(parent_concept, child_concept)] = [(relation_type, 1)]
                        else:
                            if (parent_concept, child_concept) in concepts_relations_dict:
                                if relation_type not in concepts_relations_dict.get((parent_concept, child_concept)):
                                    concepts_relations_dict[(parent_concept, child_concept)].append(relation_type)
                            else:
                                concepts_relations_dict[(parent_concept, child_concept)] = [relation_type]

    with open(CONCEPTS_RELATIONS_DICT, "wb") as dict_file:
        js.dump(concepts_relations_dict, dict_file)

    return concepts_relations_dict


if __name__ == "__main__":
    # extract_concepts_relations_pairs(with_relation_frequency=True)

    concepts_rels_dict = get_concepts_relations_pairs()

    multiple_rels_pairs = 0
    rels_per_pairs = []

    for item, i in zip(list(concepts_rels_dict.items()), list(range(len(list(concepts_rels_dict.items()))))):
        rels_per_pairs.append(len(item[1]))
        if len(item[1]) > 1:
            print(str(item[0]) + " | " + str(item[1]))
            multiple_rels_pairs += 1

    print("%d / %d" % (multiple_rels_pairs, len(list(concepts_rels_dict.items()))))
    print("%f" % (float(multiple_rels_pairs) / len(list(concepts_rels_dict.items()))))
    print("%f" % (np.mean(rels_per_pairs)))
    print("%f" % (np.mean([n for n in rels_per_pairs if n != 1])))
