import os
from typing import List
import logging
import dynet as dy

from data_extraction.word_embeddings_reader import read_glove_embeddings_from_file
from deep_dynet import support as ds
from deep_dynet.support import Vocab
from models.concept import IdentifiedConcepts, Concept
from models.node import Node
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.trainer_util import \
    generate_amr_node_for_predicted_parents, calculate_smatch, log_test_entry_data, plot_train_test_acc_loss, \
    construct_concept_glove_embeddings_list, ArcsTrainerHyperparameters, ArcsTrainerResultPerEpoch, \
    log_results_per_epoch
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.training_arcs_data_extractor import \
    ArcsTrainingEntry

CONCEPTS_EMBEDDING_SIZE = 128
CONCEPTS_GLOVE_EMBEDDING_SIZE = 100
CONCEPT_TAG_EMBEDDING_SIZE = 0
CONCEPT_LEMMA_EMBEDDING_SIZE = 0
LSTM_IN_DIM = CONCEPTS_EMBEDDING_SIZE + \
              CONCEPTS_GLOVE_EMBEDDING_SIZE + \
              CONCEPT_TAG_EMBEDDING_SIZE + \
              CONCEPT_LEMMA_EMBEDDING_SIZE
LSTM_OUT_DIM = 50
BILSTM_OUT_DIM = 2 * LSTM_OUT_DIM
LSTM_NO_LAYERS = 1
MLP_CONCEPT_INTERNAL_DIM = 32


class ArcsDynetGraph:
    def __init__(self, concepts_vocab, concept_glove_embeddings_list, hyperparams: ArcsTrainerHyperparameters):
        self.hyperparams = hyperparams
        self.model = dy.Model()
        self.concepts_vocab: Vocab = concepts_vocab
        self.concept_embeddings = self.model.add_lookup_parameters((concepts_vocab.size(), CONCEPTS_EMBEDDING_SIZE))
        self.glove_embeddings = self.model.add_lookup_parameters((concepts_vocab.size(), CONCEPTS_GLOVE_EMBEDDING_SIZE))
        self.glove_embeddings.init_from_array(dy.np.array(concept_glove_embeddings_list))
        self.trainer = dy.SimpleSGDTrainer(self.model)

        self.fwdRNN = dy.LSTMBuilder(LSTM_NO_LAYERS, LSTM_IN_DIM, LSTM_OUT_DIM, self.model)
        self.bwdRNN = dy.LSTMBuilder(LSTM_NO_LAYERS, LSTM_IN_DIM, LSTM_OUT_DIM, self.model)

        self.pU = self.model.add_parameters((MLP_CONCEPT_INTERNAL_DIM, BILSTM_OUT_DIM))
        self.pW = self.model.add_parameters((MLP_CONCEPT_INTERNAL_DIM, BILSTM_OUT_DIM))
        self.pV = self.model.add_parameters((1, MLP_CONCEPT_INTERNAL_DIM))


def get_all_concepts(concepts: List[IdentifiedConcepts]):
    concepts_list = []
    for identified_concepts in concepts:
        for concept in identified_concepts.ordered_concepts:
            concept_name = concept.name
            concepts_list.append(concept_name)
    # remove duplicates (necessary for using ds.Vocab.from_list)
    concepts_list = list(set(concepts_list))
    return concepts_list


# for 4 concepts (including root) it returns {1: [0,2,3], 2: [0,1,3], 3: [0,1,2]}
def get_potential_heads(no_sentence_concepts):
    sentence_concepts_indexes = range(no_sentence_concepts)
    potential_heads = {}
    for current_concept in sentence_concepts_indexes:
        if current_concept != 0:
            potential_heads[current_concept] = [c for c in sentence_concepts_indexes if c != current_concept]
    return potential_heads


def get_gold_head_index(concept_index, potential_heads_list, parent_vector):
    parent = parent_vector[concept_index]
    return potential_heads_list.index(parent)


def get_predicted_parent(potential_heads_list, predicted_index):
    parent = potential_heads_list[predicted_index]
    return parent


def get_concept_representation(arcs_graph: ArcsDynetGraph, c: Concept):
    c_index = arcs_graph.concepts_vocab.w2i[c.name]
    concept_trained_embedding = arcs_graph.concept_embeddings[c_index]
    concept_glove_embedding = dy.lookup(arcs_graph.glove_embeddings, c_index, False)
    # return dy.noise(ce, 0.1)
    return dy.concatenate([concept_trained_embedding, concept_glove_embedding])


def build_graph(arcs_graph: ArcsDynetGraph, sentence_concepts: List[Concept]):
    # renew computational graph
    dy.renew_cg()

    # initialize the RNNs
    f_init = arcs_graph.fwdRNN.initial_state()
    b_init = arcs_graph.bwdRNN.initial_state()

    # get concept embeddings
    concept_representations = [get_concept_representation(arcs_graph, c) for c in sentence_concepts]

    fw_exps = f_init.transduce(concept_representations)
    bw_exps = b_init.transduce(reversed(concept_representations))

    # biLSTM states
    bi = [dy.concatenate([f, b]) for f, b in zip(fw_exps, reversed(bw_exps))]

    # put dropout here

    potential_heads_idx = get_potential_heads(len(sentence_concepts))
    graph_outputs = {}
    for c_index, potential_heads in potential_heads_idx.items():
        outputs_list = []
        for h_index in potential_heads:
            u = dy.parameter(arcs_graph.pU)
            w = dy.parameter(arcs_graph.pW)
            v = dy.parameter(arcs_graph.pV)
            # concept a_i
            a_i = bi[c_index]
            # potential head of concept a_i
            a_j = bi[h_index]
            tanh = dy.tanh(u * a_j + w * a_i)
            tanh = dy.dropout(tanh, arcs_graph.hyperparams.mlp_dropout)
            g = v * tanh
            outputs_list.append(g)
        graph_outputs[c_index] = dy.concatenate(outputs_list)
    return graph_outputs


def train_sentence(arcs_graph: ArcsDynetGraph, identified_concepts: IdentifiedConcepts, parent_vector: List):
    graph_outputs = build_graph(arcs_graph, identified_concepts.ordered_concepts)
    potential_heads_idx = get_potential_heads(len(identified_concepts.ordered_concepts))
    concept_losses = []
    count = 0
    correct_predictions = 0
    for concept_idx, potential_heads_network_outputs in graph_outputs.items():
        gold_head_index = get_gold_head_index(concept_idx, potential_heads_idx[concept_idx], parent_vector)
        concept_loss = dy.pickneglogsoftmax(potential_heads_network_outputs, gold_head_index)
        concept_losses.append(concept_loss)
        count += 1

        # train accuracy
        out = dy.softmax(potential_heads_network_outputs)
        chosen = dy.np.argmax(out.npvalue())
        if chosen == gold_head_index:
            correct_predictions += 1

    accuracy = correct_predictions / count
    sentence_loss = dy.esum(concept_losses) / count
    scalar_sentence_loss = sentence_loss.value()
    sentence_loss.backward()
    arcs_graph.trainer.update()
    return (scalar_sentence_loss, accuracy)


def test_sentence(arcs_graph: ArcsDynetGraph, identified_concepts: IdentifiedConcepts, parent_vector: List):
    no_sentence_concepts = len(identified_concepts.ordered_concepts)
    graph_outputs = build_graph(arcs_graph, identified_concepts.ordered_concepts)
    potential_heads_idx = get_potential_heads(no_sentence_concepts)
    correct_predictions = 0
    predicted_parents = []
    for concept_idx, potential_heads_network_outputs in graph_outputs.items():
        gold_head_index = get_gold_head_index(concept_idx, potential_heads_idx[concept_idx], parent_vector)
        out = dy.softmax(potential_heads_network_outputs)
        chosen = dy.np.argmax(out.npvalue())
        if chosen == gold_head_index:
            correct_predictions += 1
        predicted_parents.append(get_predicted_parent(potential_heads_idx[concept_idx], chosen))
    # remove 1 due to ROOT concept
    total_predictions = no_sentence_concepts - 1
    accuracy = correct_predictions / total_predictions
    return (predicted_parents, accuracy)


detail_logs_file_name = "logs/detailed_logs.txt"
overview_logs_file_name = "logs/overview_logs.txt"


def train_and_test(train_entries, test_entries, relation_dict, hyperparams: ArcsTrainerHyperparameters):
    # setup logging
    if not os.path.exists('logs'):
        os.makedirs('logs')

    overview_logger = logging.getLogger('overview_logs')
    overview_logger.setLevel(logging.INFO)
    overview_logger.addHandler(logging.FileHandler('logs/overview_logs.log', 'w'))

    detail_logger = logging.getLogger('detail_logs')
    detail_logger.setLevel(logging.INFO)
    detail_logger.addHandler(logging.FileHandler('logs/detail_logs.log', 'w'))

    train_concepts = [train_entry.identified_concepts for train_entry in train_entries]
    dev_concepts = [test_entry.identified_concepts for test_entry in test_entries]
    all_concept_names = get_all_concepts(train_concepts + dev_concepts)
    all_concepts_vocab = ds.Vocab.from_list(all_concept_names)
    word_glove_embeddings = read_glove_embeddings_from_file(CONCEPTS_GLOVE_EMBEDDING_SIZE)
    concept_glove_embeddings_list = construct_concept_glove_embeddings_list(word_glove_embeddings,
                                                                            CONCEPTS_GLOVE_EMBEDDING_SIZE,
                                                                            all_concepts_vocab)
    no_train_entries = len(train_entries)
    no_test_entries = len(test_entries)

    # init plotting data
    results_per_epoch = {}

    arcs_graph = ArcsDynetGraph(all_concepts_vocab, concept_glove_embeddings_list, hyperparams)
    no_epochs = hyperparams.no_epochs
    for epoch in range(1, no_epochs + 1):
        print("Epoch " + str(epoch))
        overview_logger.info("Epoch " + str(epoch))
        # train
        sum_loss = 0
        train_entry: ArcsTrainingEntry
        sum_train_accuracy = 0
        for train_entry in train_entries:
            (entry_loss, train_entry_accuracy) = train_sentence(arcs_graph,
                                                                train_entry.identified_concepts,
                                                                train_entry.parent_vector)
            sum_loss += entry_loss
            sum_train_accuracy += train_entry_accuracy
        avg_loss = sum_loss / no_train_entries
        avg_train_accuracy = sum_train_accuracy / no_train_entries
        overview_logger.info("Loss " + str(avg_loss))
        overview_logger.info("Training accuracy " + str(avg_train_accuracy))
        # test
        sum_accuracy = 0
        sum_smatch = 0
        test_entry: ArcsTrainingEntry
        for test_entry in test_entries:
            (predicted_parents, entry_accuracy) = test_sentence(arcs_graph, test_entry.identified_concepts,
                                                                test_entry.parent_vector)
            sum_accuracy += entry_accuracy
            # amr str
            predicted_parents.insert(0, -1)
            predicted_amr_node: Node = generate_amr_node_for_predicted_parents(test_entry.identified_concepts,
                                                                               predicted_parents,
                                                                               relation_dict)
            smatch_f_score = 0
            if predicted_amr_node is not None:
                predicted_amr_str = predicted_amr_node.amr_print()
                smatch_f_score = calculate_smatch(predicted_amr_str, test_entry.amr_str)
            else:
                predicted_amr_str = "INVALID AMR"
            # logging
            log_test_entry_data(detail_logger, test_entry, entry_accuracy, smatch_f_score, predicted_parents,
                                predicted_amr_str)
            sum_smatch += smatch_f_score
        avg_accuracy = sum_accuracy / no_test_entries
        avg_smatch = sum_smatch / no_test_entries
        overview_logger.info("Test accuracy " + str(avg_accuracy))
        overview_logger.info("Avg smatch " + str(avg_smatch) + '\n')
        epoch_result = ArcsTrainerResultPerEpoch(avg_loss,
                                                 avg_train_accuracy,
                                                 avg_accuracy,
                                                 avg_smatch)
        results_per_epoch[epoch] = epoch_result
        log_results_per_epoch(overview_logger, epoch, epoch_result)
    print("Done")
    return results_per_epoch
