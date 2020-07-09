import os
import pickle
from copy import deepcopy
from typing import List
import logging
import dynet as dy
# from pymagnitude import Magnitude

from data_extraction.word_embeddings_reader import read_glove_embeddings_from_file
from deep_dynet import support as ds
from deep_dynet.support import Vocab
from definitions import PROJECT_ROOT_DIR
from models.concept import IdentifiedConcepts, Concept
from models.node import Node
from pre_post_processing.standford_pre_post_processing import post_processing_on_parent_vector
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.arcs_trainer_util import \
    calculate_smatch, log_test_entry_data, \
    construct_ordered_concepts_embeddings_list, ArcsTrainerHyperparameters, ArcsTrainerResultPerEpoch, \
    log_results_per_epoch, generate_amr_node_for_vector_of_parents, construct_concept_glove_embeddings_list
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.training_arcs_data_extractor import \
    ArcsTrainingEntry, read_train_test_data, ArcsTraingAndTestData

CONCEPT_TAG_EMBEDDING_SIZE = 0
CONCEPT_LEMMA_EMBEDDING_SIZE = 0
FASTTEXT_DIM = 300


class ArcsDynetGraph:
    def __init__(self, concepts_vocab, hyperparams: ArcsTrainerHyperparameters, model=None):
        self.hyperparams = hyperparams
        if model is None:
            global_model = dy.Model()
        else:
            global_model = model
        self.model = global_model.add_subcollection("arcs")

        # embeddings
        self.concepts_vocab: Vocab = concepts_vocab
        # glove
        if hyperparams.glove_embeddings_size != 0:
            self.glove_embeddings = self.model.add_lookup_parameters(
                (concepts_vocab.size(), hyperparams.glove_embeddings_size))
            # intialize glove embeddings
            word_glove_embeddings = read_glove_embeddings_from_file(hyperparams.glove_embeddings_size)
            concept_glove_embeddings_list = construct_concept_glove_embeddings_list(word_glove_embeddings,
                                                                                    hyperparams.glove_embeddings_size,
                                                                                    concepts_vocab)
            self.glove_embeddings.init_from_array(dy.np.array(concept_glove_embeddings_list))

        # trainable embeddings
        self.concept_embeddings = self.model.add_lookup_parameters(
            (concepts_vocab.size(), hyperparams.trainable_embeddings_size))

        # lstms
        lstm_in_dim = hyperparams.glove_embeddings_size + hyperparams.trainable_embeddings_size + \
                      + CONCEPT_TAG_EMBEDDING_SIZE + CONCEPT_LEMMA_EMBEDDING_SIZE
        self.fwdRNN = dy.LSTMBuilder(hyperparams.no_lstm_layers,
                                     lstm_in_dim, hyperparams.lstm_out_dim,
                                     self.model)
        self.bwdRNN = dy.LSTMBuilder(hyperparams.no_lstm_layers,
                                     lstm_in_dim, hyperparams.lstm_out_dim,
                                     self.model)

        # mlp
        bilstm_out_dim = 2 * hyperparams.lstm_out_dim
        self.pU = self.model.add_parameters((hyperparams.mlp_dim, bilstm_out_dim))
        self.pW = self.model.add_parameters((hyperparams.mlp_dim, bilstm_out_dim))
        self.pV = self.model.add_parameters((1, hyperparams.mlp_dim))

        # trainer
        self.trainer = dy.SimpleSGDTrainer(self.model)


def get_all_concepts(concepts: List[IdentifiedConcepts]):
    concepts_list = []
    for identified_concepts in concepts:
        for concept in identified_concepts.ordered_concepts:
            concept_name = concept.name
            concepts_list.append(concept_name)
    # remove duplicates (necessary for using ds.Vocab.from_list)
    concepts_list = list(set(concepts_list))
    return concepts_list


def get_potential_heads(no_sentence_concepts):
    """
        for 4 concepts (including root) it returns {1: [0,2,3], 2: [0,1,3], 3: [0,1,2]}
    """
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
    embeddings = []
    concept_stripped = Concept.strip_concept_sense(c.name)
    c_index = -1
    if c.name in arcs_graph.concepts_vocab.w2i.keys():
        c_index = arcs_graph.concepts_vocab.w2i[c.name]
    # glove
    if arcs_graph.hyperparams.glove_embeddings_size != 0:
        if c_index != -1:
            concept_glove_embedding = dy.lookup(arcs_graph.glove_embeddings, c_index, False)
        else:
            # concept_glove_embedding = dy.inputTensor(arcs_graph.glove_vectors.query(concept_stripped))
            concept_glove_embedding = dy.zeros(arcs_graph.hyperparams.glove_embeddings_size)
        embeddings.append(concept_glove_embedding)
    # trained with the network
    if arcs_graph.hyperparams.trainable_embeddings_size != 0:
        if c_index != -1:
            concept_trained_embedding = arcs_graph.concept_embeddings[c_index]
        else:
            # maybe add an UNK concept instead
            concept_trained_embedding = dy.zeros(arcs_graph.hyperparams.trainable_embeddings_size, batch_size=1)
        embeddings.append(concept_trained_embedding)
    # return dy.noise(ce, 0.1)
    return dy.concatenate(embeddings)


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


def train_or_test_for_parent_vector(arcs_graph: ArcsDynetGraph, identified_concepts: IdentifiedConcepts,
                                    parent_vector: List[int],
                                    isTrain: bool):
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

    parent_vector_accuracy = correct_predictions / count
    parent_vector_loss = dy.esum(concept_losses) / count
    parent_vector_loss_value = parent_vector_loss.value()
    if isTrain:
        parent_vector_loss.backward()
        arcs_graph.trainer.update()
    return parent_vector_loss_value, parent_vector_accuracy


def predict_vector_of_parents(arcs_graph: ArcsDynetGraph,
                              ordered_concepts: List[Concept],
                              hyperparams: ArcsTrainerHyperparameters):
    no_sentence_concepts = len(ordered_concepts)
    graph_outputs = build_graph(arcs_graph, ordered_concepts)
    potential_heads_idx = get_potential_heads(no_sentence_concepts)
    predicted_vector_of_parents = []
    for concept_idx, potential_heads_network_outputs in graph_outputs.items():
        out = dy.softmax(potential_heads_network_outputs)
        output_values = out.npvalue()
        # normalize outputs so that max -> 1
        normalized_values = []
        maxval = max(output_values)
        for output_value in output_values:
            normalized_values.append(output_value / maxval)
        parents_for_concept = []
        for i in range(0, len(normalized_values)):
            if normalized_values[i] >= hyperparams.reentrancy_threshold:
                parents_for_concept.append(get_predicted_parent(potential_heads_idx[concept_idx], i))
        predicted_vector_of_parents.append(parents_for_concept)
    return predicted_vector_of_parents


def train_amr(arcs_graph: ArcsDynetGraph, identified_concepts: IdentifiedConcepts, parent_vectors: List):
    loss_per_amr_sum = 0
    train_acc_per_amr_sum = 0
    for parent_vector in parent_vectors:
        parent_vector_loss_value, parent_vector_accuracy = train_or_test_for_parent_vector(arcs_graph,
                                                                                           identified_concepts,
                                                                                           parent_vector,
                                                                                           True)
        loss_per_amr_sum += parent_vector_loss_value
        train_acc_per_amr_sum += parent_vector_accuracy
    no_parent_vectors = len(parent_vectors)
    loss_per_amr = loss_per_amr_sum / no_parent_vectors
    train_acc_per_amr = train_acc_per_amr_sum / no_parent_vectors
    return loss_per_amr, train_acc_per_amr


def test_amr(arcs_graph: ArcsDynetGraph,
             test_entry: ArcsTrainingEntry,
             hyperparams: ArcsTrainerHyperparameters,
             relation_dict, detail_logger):
    sum_loss = 0
    sum_accuracy = 0
    for parent_vector in test_entry.parent_vectors:
        loss, acc = train_or_test_for_parent_vector(arcs_graph, test_entry.identified_concepts, parent_vector, False)
        sum_loss += loss
        sum_accuracy += acc
    loss_per_amr = sum_loss / len(test_entry.parent_vectors)
    accuracy_per_amr = sum_accuracy / len(test_entry.parent_vectors)
    smatch_f_score = 0

    # necessary so that the identidied concepts are not postprocessed for the next iteration
    identified_concepts_copy = deepcopy(test_entry.identified_concepts)
    # predict vector of parents for amr
    predicted_vector_of_parents = predict_vector_of_parents(arcs_graph,
                                                            test_entry.identified_concepts.ordered_concepts,
                                                            hyperparams)
    # for the fake root
    predicted_vector_of_parents.insert(0, [-1])
    # add postprocessing
    if hyperparams.use_preprocessing:
        post_processing_on_parent_vector(identified_concepts_copy,
                                         predicted_vector_of_parents,
                                         test_entry.preprocessed_sentence,
                                         test_entry.preprocessing_metadata)
    predicted_amr_node: Node = generate_amr_node_for_vector_of_parents(identified_concepts_copy,
                                                                       predicted_vector_of_parents,
                                                                       relation_dict)
    if predicted_amr_node is not None:
        predicted_amr_str = predicted_amr_node.amr_print_with_reentrancy()
        gold_amr_str = test_entry.amr_str
        smatch_f_score = calculate_smatch(predicted_amr_str, gold_amr_str)
        valid_amr = 1
    else:
        predicted_amr_str = "INVALID AMR"
    # logging
    if detail_logger is not None:
        log_test_entry_data(detail_logger,
                            test_entry, accuracy_per_amr, smatch_f_score, loss_per_amr,
                            predicted_vector_of_parents,
                            predicted_amr_str)
    return loss_per_amr, accuracy_per_amr, smatch_f_score


def train(arcs_graph: ArcsDynetGraph, train_and_test_data: ArcsTraingAndTestData, overview_logger):
    sum_loss = 0
    train_entry: ArcsTrainingEntry
    sum_train_accuracy = 0
    for train_entry in train_and_test_data.train_entries:
        loss_per_amr_sum, train_acc_per_amr_sum = train_amr(arcs_graph,
                                                            train_entry.identified_concepts,
                                                            train_entry.parent_vectors)
        sum_loss += loss_per_amr_sum
        sum_train_accuracy += train_acc_per_amr_sum
    avg_loss = sum_loss / train_and_test_data.no_train_amrs
    avg_train_accuracy = sum_train_accuracy / train_and_test_data.no_train_amrs
    overview_logger.info("Loss " + str(avg_loss))
    overview_logger.info("Training accuracy " + str(avg_train_accuracy))
    return avg_loss, avg_train_accuracy


def test(arcs_graph: ArcsDynetGraph, train_and_test_data: ArcsTraingAndTestData,
         hyperparams: ArcsTrainerHyperparameters,
         relation_dict, overview_logger, detail_logger):
    sum_accuracy = 0
    sum_smatch = 0
    sum_loss = 0
    test_entry: ArcsTrainingEntry
    for test_entry in train_and_test_data.test_entries:
        loss_per_amr, accuracy, smatch_f_score = test_amr(arcs_graph,
                                                          test_entry,
                                                          hyperparams,
                                                          relation_dict,
                                                          detail_logger)
        sum_accuracy += accuracy
        sum_smatch += smatch_f_score
        sum_loss += loss_per_amr
    avg_accuracy = sum_accuracy / train_and_test_data.no_test_amrs
    avg_smatch = sum_smatch / train_and_test_data.no_test_amrs
    avg_test_loss = sum_loss / train_and_test_data.no_test_amrs
    overview_logger.info("Test accuracy " + str(avg_accuracy))
    overview_logger.info("Avg smatch " + str(avg_smatch) + '\n')
    return avg_test_loss, avg_accuracy, avg_smatch


def train_and_test(relation_dict, hyperparams: ArcsTrainerHyperparameters):
    # setup logging
    if not os.path.exists('logs'):
        os.makedirs('logs')

    overview_logger = logging.getLogger('overview_logs')
    overview_logger.setLevel(logging.INFO)
    overview_logger.addHandler(logging.FileHandler('logs/overview_logs.log', 'w'))

    detail_logger = logging.getLogger('detail_logs')
    detail_logger.setLevel(logging.INFO)
    detail_logger.addHandler(logging.FileHandler('logs/detail_logs.log', 'w'))

    train_and_test_data: ArcsTraingAndTestData = read_train_test_data(hyperparams.unaligned_tolerance,
                                                                      hyperparams.max_sen_len,
                                                                      hyperparams.max_parents_vectors,
                                                                      hyperparams.use_preprocessing,
                                                                      hyperparams.alignment)
    train_concepts = [train_entry.identified_concepts for train_entry in train_and_test_data.train_entries]
    dev_concepts = [test_entry.identified_concepts for test_entry in train_and_test_data.test_entries]
    # get only concepts from train
    # all_concept_names = get_all_concepts(train_concepts + dev_concepts)
    all_concept_names = get_all_concepts(train_concepts)
    all_concepts_vocab = ds.Vocab.from_list(all_concept_names)

    # init plotting data
    results_per_epoch = {}

    arcs_graph = ArcsDynetGraph(all_concepts_vocab, hyperparams)
    for epoch in range(1, hyperparams.no_epochs + 1):
        print("Epoch " + str(epoch))
        overview_logger.info("Epoch " + str(epoch))
        # train
        avg_loss, avg_train_accuracy = train(arcs_graph, train_and_test_data, overview_logger)
        # test
        avg_test_loss, avg_accuracy, avg_smatch = test(arcs_graph, train_and_test_data,
                                                       hyperparams,
                                                       relation_dict, overview_logger,
                                                       detail_logger)
        epoch_result = ArcsTrainerResultPerEpoch(avg_loss,
                                                 avg_train_accuracy,
                                                 avg_test_loss,
                                                 avg_accuracy,
                                                 avg_smatch)
        results_per_epoch[epoch] = epoch_result
        log_results_per_epoch(overview_logger, epoch, epoch_result)

    # save model
    model_dir = 'arcs_models/' + str(hyperparams)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(model_dir + "/all_concepts_vocab", "wb") as f:
        pickle.dump(all_concepts_vocab, f)

    # save model
    arcs_graph.model.save(model_dir + "/graph")
    print("Done")
    return results_per_epoch


ARC_MODELS_PATH = PROJECT_ROOT_DIR + \
                  '/trainers/arcs/head_selection/head_selection_on_ordered_concepts/arcs_models/'


def save_arcs_model(arcs_graph: ArcsDynetGraph):
    # save model
    model_dir = 'arcs_models/' + str(arcs_graph.hyperparams)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(model_dir + "/all_concepts_vocab", "wb") as f:
        pickle.dump(arcs_graph.concepts_vocab, f)
    # save model
    arcs_graph.model.save(model_dir + "/graph")


def load_arcs_model(hyperparams, model=None):
    model_dir = ARC_MODELS_PATH + str(hyperparams)
    if not os.path.exists(model_dir):
        print("No such trained model" + model_dir)
        return None
    else:
        # get vocabs
        with open(model_dir + "/all_concepts_vocab", "rb") as f:
            all_concepts_vocab = pickle.load(f)
        # create graph
        arcs_graph = ArcsDynetGraph(all_concepts_vocab, hyperparams, model)
        # populate
        arcs_graph.model.populate(model_dir + "/graph")
        return arcs_graph
