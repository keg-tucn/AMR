from typing import List

import dynet as dy

from deep_dynet import support as ds
from deep_dynet.support import Vocab
from models.concept import IdentifiedConcepts, Concept
from models.node import Node
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.trainer_util import get_all_paths, \
    generate_amr_node_for_predicted_parents
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.training_arcs_data_extractor import \
    generate_arcs_training_data, ArcsTrainingEntry
from trainers.arcs.head_selection.relations_dictionary_extractor import extract_relation_dict

CONCEPTS_EMBEDDING_SIZE = 128
LSTM_IN_DIM = 128
LSTM_OUT_DIM = 50
BILSTM_OUT_DIM = 2 * 50
LSTM_NO_LAYERS = 1
MLP_CONCEPT_INTERNAL_DIM = 32


class ArcsDynetGraph:
    def __init__(self, concepts_vocab):
        self.model = dy.Model()
        self.concepts_vocab: Vocab = concepts_vocab
        self.concept_embeddings = self.model.add_lookup_parameters((concepts_vocab.size(), CONCEPTS_EMBEDDING_SIZE))
        self.trainer = dy.SimpleSGDTrainer(self.model)

        self.fwdRNN = dy.LSTMBuilder(LSTM_NO_LAYERS, LSTM_IN_DIM, LSTM_OUT_DIM, self.model)
        self.bwdRNN = dy.LSTMBuilder(LSTM_NO_LAYERS, LSTM_IN_DIM, LSTM_OUT_DIM, self.model)

        self.pU = self.model.add_parameters((MLP_CONCEPT_INTERNAL_DIM, BILSTM_OUT_DIM))
        self.pW = self.model.add_parameters((MLP_CONCEPT_INTERNAL_DIM, BILSTM_OUT_DIM))
        self.pV = self.model.add_parameters((1, MLP_CONCEPT_INTERNAL_DIM))


def create_concepts_vocab(train_concepts: List[IdentifiedConcepts], test_concepts: List[IdentifiedConcepts]):
    concepts_list = []
    train_and_test_concepts = train_concepts + test_concepts
    for identified_concepts in train_and_test_concepts:
        for concept in identified_concepts.ordered_concepts:
            concept_name = concept.name
            concepts_list.append(concept_name)
    # remove duplicates (necessary for using ds.Vocab.from_list)
    concepts_list = list(set(concepts_list))
    return ds.Vocab.from_list(concepts_list)


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
    # TODO: do something in case the concept is not in the dictionary
    # TODO: treat AMRs with multiple nodes with the same concept
    #  1) use the variable in the representation? -> might not make sense, variable has no info
    #  2) use positional information -> maybe the positional information is already given by the bilstm state
    #  3) use more info from sentence (need alignment at this stage)
    c_index = arcs_graph.concepts_vocab.w2i[c.name]
    ce = arcs_graph.concept_embeddings[c_index]
    return dy.noise(ce, 0.1)


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

    # test graph ok till here (do a forward)
    bivalues = [bi_entry.value() for bi_entry in bi]

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
            g = v * dy.tanh(u * a_j + w * a_i)
            outputs_list.append(g)
        graph_outputs[c_index] = dy.concatenate(outputs_list)
    return graph_outputs


def train_sentence(arcs_graph: ArcsDynetGraph, identified_concepts: IdentifiedConcepts, parent_vector: List):
    graph_outputs = build_graph(arcs_graph, identified_concepts.ordered_concepts)
    potential_heads_idx = get_potential_heads(len(identified_concepts.ordered_concepts))
    concept_losses = []
    count = 0
    for concept_idx, potential_heads_network_outputs in graph_outputs.items():
        gold_head_index = get_gold_head_index(concept_idx, potential_heads_idx[concept_idx], parent_vector)
        # TODO: could I still use pickneglogsoftmax? maybe create an expression out of a list of expressions
        concept_loss = dy.pickneglogsoftmax(potential_heads_network_outputs, gold_head_index)
        concept_losses.append(concept_loss)
        count += 1
    sentence_loss = dy.esum(concept_losses) / count
    scalar_sentence_loss = sentence_loss.value()
    sentence_loss.backward()
    arcs_graph.trainer.update()
    return scalar_sentence_loss


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
        predicted_parents.append(get_predicted_parent(potential_heads_idx[concept_idx],chosen))
    # remove 1 due to ROOT concept
    total_predictions = no_sentence_concepts - 1
    accuracy = correct_predictions / total_predictions
    return (predicted_parents, accuracy)



def read_train_test_data():
    train_entries, no_train_failed = generate_arcs_training_data(get_all_paths('training'))
    no_train_entries = len(train_entries)
    print(str(no_train_entries) + ' train entries processed ' + str(no_train_failed) + ' train entries failed')
    test_entries, no_test_failed = generate_arcs_training_data(get_all_paths('dev'))
    no_test_entries = len(test_entries)
    print(str(no_test_entries) + ' test entries processed ' + str(no_test_failed) + ' test entries failed')
    return (train_entries, no_train_entries, test_entries, no_test_entries)


loggfile = "logs/log.txt"

if __name__ == "__main__":
    # TODO: train_concepts and test_concepts
    (train_entries, no_train_entries, test_entries, no_test_entries) = read_train_test_data()
    relation_dict = extract_relation_dict(get_all_paths('training'))
    train_concepts = [train_entry.identified_concepts for train_entry in train_entries]
    dev_concepts = [test_entry.identified_concepts for test_entry in test_entries]
    all_concepts_vocab = create_concepts_vocab(train_concepts, dev_concepts)

    # prepare logging file
    f = open(loggfile, "w")

    arcs_graph = ArcsDynetGraph(all_concepts_vocab)
    for epoch in range(2):
        print("Epoch " + str(epoch))
        f.write("Epoch " + str(epoch)+'\n')
        # train
        sum_loss = 0
        train_entry: ArcsTrainingEntry
        for train_entry in train_entries:
            entry_loss = train_sentence(arcs_graph, train_entry.identified_concepts, train_entry.parent_vector)
            sum_loss += entry_loss
        avg_loss = sum_loss / no_train_entries
        print("Loss " + str(avg_loss))
        # test
        sum_accuracy = 0
        test_entry: ArcsTrainingEntry
        for test_entry in test_entries:
            (predicted_parents, entry_accuracy) = test_sentence(arcs_graph, test_entry.identified_concepts, test_entry.parent_vector)
            sum_accuracy += entry_accuracy
            # amr str
            predicted_parents.insert(0,-1)
            amr_node: Node = generate_amr_node_for_predicted_parents(test_entry.identified_concepts,
                                                             predicted_parents,
                                                             relation_dict)
            if amr_node is not None:
                amr_str = amr_node.amr_print()
            else:
                amr_str = "INVALID AMR"
            # logging
            f.write('Entry accuracy: '+str(entry_accuracy)+'\n')
            f.write('Predicted parents: '+str(predicted_parents)+'\n')
            f.write('Predicted amr:\n'+amr_str+'\n')
            f.write(test_entry.logging_info)
        avg_accuracy = sum_accuracy / no_test_entries
        print("Accuracy " + str(avg_accuracy))
        print()
    print("Done")
    f.close()
