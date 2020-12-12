import string
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import random

from deep_dynet.support import Vocab
from models.concept import IdentifiedConcepts, Concept
from models.node import Node
from smatch import smatch_util, smatch_amr
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.training_arcs_data_extractor import \
    ArcsTrainingEntry
from trainers.arcs.head_selection.relations_dictionary_extractor import get_relation_between_concepts

Adam_Trainer = 'Adam'
SGD_Trainer = 'SGD'
UNKNOWN_CHAR = '<unk-char>'
class ArcsTrainerHyperparameters:
    def __init__(self, no_epochs, mlp_dropout,
                 unaligned_tolerance,
                 max_sen_len,
                 max_parents_vectors,
                 reentrancy_threshold,
                 use_preprocessing: bool,
                 trainable_embeddings_size: int,
                 glove_embeddings_size: int,
                 lstm_out_dim: int,
                 mlp_dim: int,
                 no_lstm_layers: int,
                 alignment: str,
                 experimental_run: bool,
                 two_char_rnns: bool,
                 glove0: bool,
                 char_cnn_cutoff: int,
                 use_verb_flag: bool,
                 trainer):
        self.no_epochs = no_epochs
        self.mlp_dropout = mlp_dropout
        # how many concepts with no alignment we allow in the ordered concepts (percentage: 0-none,1-all)
        self.unaligned_tolerance = unaligned_tolerance
        self.max_sen_len = max_sen_len
        self.max_parents_vectors = max_parents_vectors
        self.reentrancy_threshold = reentrancy_threshold
        self.use_preprocessing = use_preprocessing
        self.trainable_embeddings_size = trainable_embeddings_size
        self.glove_embeddings_size = glove_embeddings_size
        self.lstm_out_dim = lstm_out_dim
        self.mlp_dim = mlp_dim
        self.no_lstm_layers = no_lstm_layers
        self.alignment = alignment
        self.experimental_run = experimental_run
        self.two_char_rnns = two_char_rnns
        self.glove0 = glove0
        self.char_cnn_cutoff = char_cnn_cutoff
        self.use_verb_flag = use_verb_flag
        self.trainer = trainer

    def __str__(self):
        name = self.alignment + \
               '_ep_' + str(self.no_epochs) + \
               '_mdrop_' + str(self.mlp_dropout) + \
               '_unaltol_' + str(self.unaligned_tolerance) + \
               '_sl_' + str(self.max_sen_len) + \
               '_pv_' + str(self.max_parents_vectors) + \
               '_th_' + str(self.reentrancy_threshold) + \
               '_prep_' + str(self.use_preprocessing) + \
               '_tEmb_' + str(self.trainable_embeddings_size) + \
               '_gEmb_' + str(self.glove_embeddings_size) + \
               '_dims_' + str(self.lstm_out_dim) + '-' + str(self.mlp_dim) + \
               '_l_' + str(self.no_lstm_layers) + \
               '_exp_' + str(self.experimental_run) + \
               '_2cnn_'+str(self.two_char_rnns) +\
               '_glv0_'+str(self.glove0) +\
               '_cnnth_' + str(self.char_cnn_cutoff) \
               +'_vb_'+str(self.use_verb_flag)
        if self.trainer!=SGD_Trainer:
            name = name +'_'+self.trainer
        return name

    def __repr__(self):
        name = self.alignment + \
               '_ep_' + str(self.no_epochs) + \
               '_mdrop_' + str(self.mlp_dropout) + \
               '_unaltol_' + str(self.unaligned_tolerance) + \
               '_sl_' + str(self.max_sen_len) + \
               '_pv_' + str(self.max_parents_vectors) + \
               '_th_' + str(self.reentrancy_threshold) + \
               '_prep_' + str(self.use_preprocessing) + \
               '_tEmb_' + str(self.trainable_embeddings_size) + \
               '_gEmb_' + str(self.glove_embeddings_size) + \
               '_dims_' + str(self.lstm_out_dim) + '-' + str(self.mlp_dim) + \
               '_l_' + str(self.no_lstm_layers) + \
               '_exp_' + str(self.experimental_run) + \
               '_2cnn_'+str(self.two_char_rnns) +\
               '_glv0_'+str(self.glove0) +\
               '_cnnth_' + str(self.char_cnn_cutoff) \
               +'_vb_'+str(self.use_verb_flag)
        if self.trainer!=SGD_Trainer:
            name = name +'_'+self.trainer
        return name



class ArcsTrainerResultPerEpoch:
    def __init__(self, avg_loss,
                 avg_train_accuracy,
                 avg_test_loss,
                 avg_test_accuracy,
                 avg_smatch):
        self.avg_loss = avg_loss
        self.avg_train_accuracy = avg_train_accuracy
        self.avg_test_loss = avg_test_loss
        self.avg_test_accuracy = avg_test_accuracy
        self.avg_smatch = avg_smatch


def is_verb(concept):
    splitted_concept = concept.split('-')
    if splitted_concept[len(splitted_concept) - 1].isdigit():
        return 1
    return 0

def construct_word_freq_dict(word_list: List[str]):
    freq_table = {}
    for word in word_list:
        if word not in freq_table.keys():
            freq_table[word] = 0
        freq_table[word] += 1
    return freq_table


def write_results_per_epoch(file, epoch_no, result: ArcsTrainerResultPerEpoch):
    file.write("Epoch " + str(epoch_no)+'\n')
    file.write("Loss " + str(result.avg_loss)+'\n')
    file.write("Training accuracy " + str(result.avg_train_accuracy)+'\n')
    file.write("Test Loss " + str(result.avg_test_loss)+'\n')
    file.write("Test accuracy " + str(result.avg_test_accuracy)+'\n')
    file.write("Avg smatch " + str(result.avg_smatch) + '\n\n')


def log_results_per_epoch(logger, epoch_no, result: ArcsTrainerResultPerEpoch):
    logger.info("Epoch " + str(epoch_no))
    logger.info("Loss " + str(result.avg_loss))
    logger.info("Training accuracy " + str(result.avg_train_accuracy))
    logger.info("Test Loss " + str(result.avg_test_loss))
    logger.info("Test accuracy " + str(result.avg_test_accuracy))
    logger.info("Avg smatch " + str(result.avg_smatch) + '\n')


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


def generate_amr_node_for_vector_of_parents(identified_concepts: IdentifiedConcepts,
                                            vector_of_parents: List[List[int]],
                                            relations_dict):
    """ generate amr (type Node) from identified concepts, parents and relations dict
        in the vector of parents, a node can have multiple parents
            If the generated AMR does not have a root, pick a random node as root
    """
    # create list of nodes
    # the parent should not be themselves
    for i in range(0, len(vector_of_parents)):
        if i in vector_of_parents[i]:
            print('happened at index i' + str(i))
            raise RuntimeError('This should never happen')
            return None

    nodes: List[Node] = []
    for i in range(0, len(identified_concepts.ordered_concepts)):
        concept = identified_concepts.ordered_concepts[i]
        # TODO: better condition for this
        # I will not have a concept variable at test time :O How do I solve this?
        if concept.variable == concept.name and concept.variable != 'i':
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
    for i in range(1, len(vector_of_parents)):
        for parent in vector_of_parents[i]:
            if parent == 0:
                # ROOT
                root = nodes[i]
            else:
                parent = nodes[parent]
                child = nodes[i]
                relation = get_relation_between_nodes(relations_dict, parent, child)
                parent.add_child(child, relation)

    if root is None:
        # get a random Node to be root (maybe should get a node with most descendents, but fine for now)
        # start from 1 because of the fake root
        root_idx = random.randint(1, len(nodes) - 1)
        return nodes[root_idx]
    return root


def generate_amr_node_for_parents_vector(identified_concepts: IdentifiedConcepts,
                                         parent_vector: List[int],
                                         relations_dict):
    """ generate amr (type Node) from identified concepts, parents and relations dict
    If the generated amr is not a valid AMR (no root, cycles), return none
    """

    # chek amr validity
    if not is_valid_amr(parent_vector):
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
    for i in range(1, len(parent_vector)):
        if parent_vector[i] == 0:
            # ROOT
            root = nodes[i]
        else:
            parent = nodes[parent_vector[i]]
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


def log_test_entry_data(logger, test_entry: ArcsTrainingEntry,
                        entry_accuracy: float,
                        smatch_f_score: float,
                        loss: float,
                        predicted_parents: List[int],
                        predicted_amr_str: str):
    logger.info('Entry accuracy: ' + str(entry_accuracy))
    logger.info('Smatch: ' + str(smatch_f_score))
    logger.info('Loss: ' + str(loss))
    logger.info('Predicted parents: ' + str(predicted_parents))
    logger.info('Predicted amr:\n' + predicted_amr_str)
    logger.info(test_entry.logging_info)


def plot_losses(filename: str, plotting_data):
    """
    Plot on the x axis the epoch number
    Plot on the y axis:
        the loss
        the test loss
    Takes as input plotting_data, a dictionary of the form epoch_no: ArcsTrainerResultPerEpoch
    """

    x = []
    losses = []
    test_losses = []
    for epoch_no, plot_data_entry in plotting_data.items():
        x.append(epoch_no)
        plot_data_entry: ArcsTrainerResultPerEpoch
        losses.append(plot_data_entry.avg_loss)
        test_losses.append(plot_data_entry.avg_test_loss)

    fig, ax = plt.subplots()
    ax.plot(x, losses)
    ax.plot(x, test_losses)

    ax.legend(['train_loss', 'test_loss'], loc='upper right')

    ax.set(xlabel='epoch',
           title='Losses')

    fig.savefig(filename)
    plt.show()


def plot_acc_and_smatch(filename: str, plotting_data):
    """
    Plot on the x axis the epoch number
    Plot on the y axis:
        the train accuracy
        the test accuracy
        the smatch (test)
        the percentage of valid amrs (test)
    Takes as input plotting_data, a dictionary of the form epoch_no: ArcsTrainerResultPerEpoch
    """

    x = []
    train_accuracies = []
    test_accuracies = []
    test_smatches = []
    for epoch_no, plot_data_entry in plotting_data.items():
        x.append(epoch_no)
        plot_data_entry: ArcsTrainerResultPerEpoch
        train_accuracies.append(plot_data_entry.avg_train_accuracy)
        test_accuracies.append(plot_data_entry.avg_test_accuracy)
        test_smatches.append(plot_data_entry.avg_smatch)

    fig, ax = plt.subplots()
    ax.plot(x, train_accuracies)
    ax.plot(x, test_accuracies)
    ax.plot(x, test_smatches)

    ax.legend(['train_acc', 'test_acc', 'test_smatch'], loc='upper right')

    ax.set(xlabel='epoch',
           title='Accuracies and smatch')

    fig.savefig(filename)
    plt.show()




def construct_chars_vocab():
    # add letters (lowercase and upper case) + dash + digits
    lower = list(string.ascii_lowercase)
    caps = list(string.ascii_uppercase)
    digits = [str(i) for i in range(0, 10)]
    return lower + caps + digits + ['-'] + [UNKNOWN_CHAR]


def construct_ordered_concepts_embeddings_list(magnitude_embeddings, concept_vocab: Vocab):
    """
    Create a list of embeddings for the concepts in the input concept vocab
    The list will be ordered in the order of the concept indexes in the vocab
    Uses magnitude embeddings (should treat oov)
    Input:
        magnitude_embeddings: embeddings vector
        embedding_dim: embedding dimension
        concept_vocab: vocab of concepts (concepts associated with indexes)
    Output:
        list of glove embeddings in the order of concepts from concept_vocab
    """
    embeddings_list = []
    for concept_idx in sorted(concept_vocab.i2w.keys()):
        concept_name = concept_vocab.i2w[concept_idx]
        concept_stripped = Concept.strip_concept_sense(concept_name)
        concept_embedding = magnitude_embeddings.query(concept_stripped)
        embeddings_list.append(concept_embedding)
    return embeddings_list


def construct_concept_glove_embeddings_list(glove_embeddings, embedding_dim, concept_vocab: Vocab):
    """
    Create a list of glove embeddings for the concepts in the input concept vocab
    The list will be ordered in the order of the concept indexes in the vocab
    Input:
        glove_embeddings: dictionary of word -> glove_embedding
        embedding_dim: embedding dimension
        concept_vocab: vocab of concepts (concepts associated with indexes)
    Output:
        list of glove embeddings in the order of concepts from concept_vocab
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
