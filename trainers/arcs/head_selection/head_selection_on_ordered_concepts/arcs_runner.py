import logging
import os

from deep_dynet import support as ds
from data_extraction.dataset_reading_util import get_all_paths, get_all_paths_for_alignment
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.arcs_trainer import get_all_concepts, \
    ArcsDynetGraph, train, test, save_arcs_model, load_arcs_model, get_all_concepts_with_duplicates
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.arcs_trainer_util import \
    ArcsTrainerHyperparameters, construct_word_freq_dict
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.baseline import MLP_DROPOUT, REENTRANCY_THRESHOLD, \
    PREPROCESSING, MAX_PARENT_VECTORS, TRAINABLE_EMB_SIZE, GLOVE_EMB_SIZE, LSTM_OUT_DIM, MLP_DIM, NO_LSTM_LAYERS, \
    CHAR_CNN_CUTOFF, USE_VERB_FLAG, TRAINER
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.training_arcs_data_extractor import \
    ArcsTraingAndTestData, read_train_test_data, generate_arcs_training_data
from trainers.arcs.head_selection.relations_dictionary_extractor import extract_relation_dict


def train_model(hyperparams: ArcsTrainerHyperparameters):
    # setup logging
    if not os.path.exists('logs'):
        os.makedirs('logs')

    overview_logger = logging.getLogger('overview_logs')
    overview_logger.setLevel(logging.INFO)
    overview_logger.addHandler(logging.FileHandler('logs/overview_logs.log', 'w'))

    train_and_test_data: ArcsTraingAndTestData = read_train_test_data(hyperparams.unaligned_tolerance,
                                                                      hyperparams.max_sen_len,
                                                                      hyperparams.max_parents_vectors,
                                                                      hyperparams.use_preprocessing,
                                                                      hyperparams.alignment)
    # enhance the train data with the dev data
    train_and_test_data.train_entries = train_and_test_data.train_entries + train_and_test_data.test_entries
    train_concepts = [train_entry.identified_concepts for train_entry in train_and_test_data.train_entries]
    all_concept_names = get_all_concepts(train_concepts)
    all_concepts_vocab = ds.Vocab.from_list(all_concept_names)

    concepts_counts_table = construct_word_freq_dict(get_all_concepts_with_duplicates(train_concepts))
    arcs_graph = ArcsDynetGraph(all_concepts_vocab, concepts_counts_table, hyperparams)
    for epoch in range(1, hyperparams.no_epochs + 1):
        print("Epoch " + str(epoch))
        overview_logger.info("Epoch " + str(epoch))
        # train
        avg_loss, avg_train_accuracy = train(arcs_graph, train_and_test_data, overview_logger)
        print('Avg loss: ' + str(avg_loss))
        print('Avg Acc: ' + str(avg_train_accuracy))

    # save model
    save_arcs_model(arcs_graph)


def test_model(hyperparams: ArcsTrainerHyperparameters, alignment):
    # load model
    arcs_graph: ArcsDynetGraph = load_arcs_model(hyperparams)

    # read relations
    train_dev_paths = list(get_all_paths_for_alignment('training', alignment)) + \
                      list(get_all_paths_for_alignment('dev', alignment))
    relation_dict = extract_relation_dict(train_dev_paths)

    # read test data
    test_entries, no_test_failed, no_pv_hist_test = generate_arcs_training_data(
        get_all_paths_for_alignment('test', alignment),
        hyperparams.unaligned_tolerance,
        hyperparams.max_sen_len,
        hyperparams.max_parents_vectors,
        hyperparams.use_preprocessing,
        False)
    train_test_data: ArcsTraingAndTestData = ArcsTraingAndTestData(train_entries=None,
                                                                   test_entries=test_entries,
                                                                   no_train_amrs=None,
                                                                   no_test_amrs=len(test_entries))
    # setup logging (todo: move to function)
    overview_logger = logging.getLogger('overview_logs')
    overview_logger.setLevel(logging.INFO)
    overview_logger.addHandler(logging.FileHandler('logs/overview_logs.log', 'w'))

    detail_logger = logging.getLogger('detail_logs')
    detail_logger.setLevel(logging.INFO)
    detail_logger.addHandler(logging.FileHandler('logs/detail_logs.log', 'w'))

    # test
    avg_test_loss, avg_accuracy, avg_smatch = test(arcs_graph,
                                                   train_test_data,
                                                   hyperparams,
                                                   relation_dict,
                                                   overview_logger, detail_logger)
    print('Avg test loss: ' + str(avg_test_loss))
    print('Avg accuracy ' + str(avg_accuracy))
    print('Avg smatch ' + str(avg_smatch))

    print('Done')


if __name__ == "__main__":

    alignment = 'jamr'
    hyperparams = ArcsTrainerHyperparameters(no_epochs=20,
                                             mlp_dropout=MLP_DROPOUT,
                                             unaligned_tolerance=0,
                                             max_sen_len=300,
                                             max_parents_vectors=MAX_PARENT_VECTORS,
                                             reentrancy_threshold=REENTRANCY_THRESHOLD,
                                             use_preprocessing=PREPROCESSING,
                                             trainable_embeddings_size=TRAINABLE_EMB_SIZE,
                                             glove_embeddings_size=GLOVE_EMB_SIZE,
                                             lstm_out_dim=LSTM_OUT_DIM,
                                             mlp_dim=MLP_DIM,
                                             no_lstm_layers=NO_LSTM_LAYERS,
                                             alignment=alignment,
                                             experimental_run=True,
                                             two_char_rnns=False,
                                             glove0=True,
                                             char_cnn_cutoff=CHAR_CNN_CUTOFF,
                                             use_verb_flag=USE_VERB_FLAG,
                                             trainer=TRAINER)
    should_train = False
    if should_train:
        train_model(hyperparams)
    else:
        test_model(hyperparams, alignment)
