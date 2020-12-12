import os
import logging
from data_extraction.dataset_reading_util import get_all_paths
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.arcs_trainer import train_and_test
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.arcs_trainer_util import \
    ArcsTrainerHyperparameters, \
    plot_losses, plot_acc_and_smatch, write_results_per_epoch, Adam_Trainer, SGD_Trainer
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.baseline import NO_EPOCHS, MLP_DROPOUT, \
    MAX_PARENT_VECTORS, REENTRANCY_THRESHOLD, PREPROCESSING, GLOVE_EMB_SIZE, TRAINABLE_EMB_SIZE, TRAINER, USE_VERB_FLAG, \
    CHAR_CNN_CUTOFF, NO_LSTM_LAYERS, MLP_DIM, LSTM_OUT_DIM

from trainers.arcs.head_selection.relations_dictionary_extractor import extract_relation_dict


def run_experiment(relation_dict, hyperparams: ArcsTrainerHyperparameters, filename=None):
    if filename is None:
        filename = str(hyperparams)
    results_per_epoch = train_and_test(relation_dict, hyperparams)

    if not os.path.exists('plots'):
        os.makedirs('plots')
    if not os.path.exists('plots/losses'):
        os.makedirs('plots/losses')
    if not os.path.exists('plots/metrics'):
        os.makedirs('plots/metrics')
    plot_losses('plots/losses/' + filename + '.png', results_per_epoch)
    plot_acc_and_smatch('plots/metrics/' + filename + '.png', results_per_epoch)

    if not os.path.exists('results'):
        os.makedirs('results')

    results = open('results/' + filename + '.txt', 'w')
    for epoch, epoch_result in results_per_epoch.items():
        write_results_per_epoch(results, epoch, epoch_result)
    results.close()


if __name__ == "__main__":
    # load train&test data + relations dict
    relation_dict = extract_relation_dict(get_all_paths('training'))

    # hyperparams = ArcsTrainerHyperparameters(no_epochs=2,
    #                                          mlp_dropout=0.5,
    #                                          unaligned_tolerance=0)
    # run_experiment(relation_dict, hyperparams)

    hyperparams = ArcsTrainerHyperparameters(no_epochs=20,
                                             mlp_dropout=MLP_DROPOUT,
                                             unaligned_tolerance=0,
                                             max_sen_len=20,
                                             max_parents_vectors=MAX_PARENT_VECTORS,
                                             reentrancy_threshold=REENTRANCY_THRESHOLD,
                                             use_preprocessing=PREPROCESSING,
                                             trainable_embeddings_size=TRAINABLE_EMB_SIZE,
                                             glove_embeddings_size=GLOVE_EMB_SIZE,
                                             lstm_out_dim=LSTM_OUT_DIM,
                                             mlp_dim=MLP_DIM,
                                             no_lstm_layers=NO_LSTM_LAYERS,
                                             alignment='jamr',
                                             experimental_run=True,
                                             two_char_rnns=False,
                                             glove0=True,
                                             char_cnn_cutoff=CHAR_CNN_CUTOFF,
                                             use_verb_flag = USE_VERB_FLAG,
                                             trainer=TRAINER)
    # try 3 experiments, 1 char RNN, 2 char RNN with glove0 false, 2 char RNN with glove0 True
    # so basically 1 big char RNN, 1 char RNN for each emb, 1 char RNN for trained and 0 for glove
    concept_representations = [(False,None)]
    # concept_representations = [(False,None),(True,False),(True,True)]
    for two_rnns, glove0 in concept_representations:
        hyperparams.two_char_rnns = two_rnns
        hyperparams.glove0 = glove0
        for i in [9,10]:
            reen_th = i * 0.1
            hyperparams.reentrancy_threshold = reen_th
            print('Reentrancy threshold '+str(reen_th))
            run_experiment(relation_dict, hyperparams)
