from os import listdir
from typing import List

from definitions import AMR_ALIGNMENTS_SPLIT
from models.concept import IdentifiedConcepts
from models.node import Node
from trainers.arcs.head_selection.relations_dictionary_extractor import get_relation_between_concepts


def get_all_paths(split: str):
    dir_path = AMR_ALIGNMENTS_SPLIT + "/" + split
    directory_content = listdir(dir_path)
    original_corpus = sorted([x for x in directory_content if "dump" not in x])
    all_paths = []
    for file_name in original_corpus:
        original_file_path = dir_path + "/" + file_name
        print(original_file_path)
        all_paths.append(original_file_path)
    return all_paths


def get_relation_between_nodes(relations_dict, parent: Node, child: Node):
    if parent.label is None:
        c1 = parent.tag
    else:
        c1 = parent.label
    if child.label is None:
        c2 = child.tag
    else:
        c2 = child.label
    return get_relation_between_concepts(relations_dict, c1, c2)


def is_valid_amr(predicted_parents: List[int]):
    # 1) check to see if the AMR has a root
    try:
        root_index = predicted_parents.index(0)
    except ValueError:
        # root not found
        return False
    # 2) check to see if the AMR has cycles
    # go along the paths starting at each node and see if a node occurs more then once (O(n*n))
    # TODO: faster alg for cycle checking
    # no of nodes including added ROOT
    no_nodes = len(predicted_parents)
    for node in range(1, no_nodes):
        visited = []
        current = node
        while current != 0:
            current = predicted_parents[current]
            if current in visited:
                return False
            visited.append(current)
    return True


def generate_amr_node_for_predicted_parents(identified_concepts: IdentifiedConcepts,
                                            predicted_parents: List[int],
                                            relations_dict):
    """ generate amr (type Node) from identified concepts, parents and relations dict
    If the generated amr is not a valid AMR (no root, cycles), return none
    """

    # chek amr validity
    if not is_valid_amr(predicted_parents):
        return None

    # create list of nodes
    nodes: List[Node] = []
    for i in range(0, len(identified_concepts.ordered_concepts)):
        concept = identified_concepts.ordered_concepts[i]
        if concept.variable == concept.name:
            # literals, interogative, -
            # TODO: investigate if there are more exceptions and maybe add a type in concept
            exceptions = ["-", "interrogative"]
            if concept.variable in exceptions:
                node: Node = Node(concept.name)
            else:
                node: Node = Node(None, concept.name)
        else:
            node: Node = Node(concept.name)
        nodes.append(node)

    root: Node = None
    # start from 1 as on 0 there is ROOT
    for i in range(1, len(predicted_parents)):
        if predicted_parents[i] == 0:
            # ROOT
            root = nodes[i]
        else:
            parent = nodes[predicted_parents[i]]
            child = nodes[i]
            relation = get_relation_between_nodes(relations_dict, parent, child)
            parent.add_child(child, relation)

    return root
