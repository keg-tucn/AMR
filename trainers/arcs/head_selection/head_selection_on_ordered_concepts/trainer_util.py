from os import listdir
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from deep_dynet.support import Vocab
from definitions import AMR_ALIGNMENTS_SPLIT
from models.concept import IdentifiedConcepts, Concept
from models.node import Node
from smatch import smatch_util, smatch_amr
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.training_arcs_data_extractor import \
    ArcsTrainingEntry
from trainers.arcs.head_selection.relations_dictionary_extractor import get_relation_between_concepts

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


def calculate_smatch(predicted_amr_str: str, gold_amr_str):
    smatch_results = smatch_util.SmatchAccumulator()
    predicted_amr_smatch = smatch_amr.AMR.parse_AMR_line(predicted_amr_str)
    gold_amr_smatch = smatch_amr.AMR.parse_AMR_line(gold_amr_str)
    smatch_f_score = smatch_results.compute_and_add(predicted_amr_smatch, gold_amr_smatch)
    return smatch_f_score


def log_test_entry_data(file, test_entry: ArcsTrainingEntry,
                        entry_accuracy: float,
                        smatch_f_score: float,
                        predicted_parents: List[int],
                        predicted_amr_str: str):
    file.write('Entry accuracy: ' + str(entry_accuracy) + '\n')
    file.write('Smatch: ' + str(smatch_f_score) + '\n')
    file.write('Predicted parents: ' + str(predicted_parents) + '\n')
    file.write('Predicted amr:\n' + predicted_amr_str + '\n')
    file.write(test_entry.logging_info)


def plot_train_test_acc_loss(plotting_data):
    """
    Plot on the x axis the epoch number
    Plot on the y axis:
        the loss
        the train accuracy
        the test accuracy
    Takes as input plotting_data, a dictionary of the form epoch_no: (loss, train_acc, test_acc)
    """

    x = []
    losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch_no, plot_data_entry in plotting_data.items():
        x.append(epoch_no)
        losses.append(plot_data_entry[0])
        train_accuracies.append(plot_data_entry[1])
        test_accuracies.append(plot_data_entry[2])

    fig, ax = plt.subplots()
    ax.plot(x, losses)
    ax.plot(x, train_accuracies)
    ax.plot(x, test_accuracies)

    ax.legend(['loss', 'train_acc', 'test_acc'], loc='upper right')

    ax.set(xlabel='epoch',
           title='Loss and accuracies')

    fig.savefig("plots/test.png")
    plt.show()


def construct_concept_glove_embeddings_list(glove_embeddings, embedding_dim, concept_vocab: Vocab):
    """
    Create a list of glove embeddings for the concepts in the input concept vocab
    The list will be ordered in the order of the concept indexes in the vocab
    Input:
        glove_embeddings: dictionary of word -> glove_embedding
        embedding_dim: embedding dimension
        concept_vocab: vocab of concepts (concepts associated with indexes)
    """
    concept_glove_embeddings_list = []
    null_embedding = np.zeros(embedding_dim)
    for concept_idx in sorted(concept_vocab.i2w.keys()):
        concept_name = concept_vocab.i2w[concept_idx]
        concept_stripped = Concept.strip_concept_sense(concept_name)
        concept_glove_embedding = glove_embeddings.get(concept_stripped)
        if concept_glove_embedding is None:
            concept_glove_embedding = null_embedding
        concept_glove_embeddings_list.append(concept_glove_embedding)
    return concept_glove_embeddings_list
