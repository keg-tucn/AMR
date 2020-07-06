import os
import logging
from data_extraction.dataset_reading_util import get_all_paths
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.trainer import train_and_test
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.trainer_util import ArcsTrainerHyperparameters, \
    plot_losses, plot_acc_and_smatch, log_results_per_epoch

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

    results_logger = logging.getLogger('results_logs')
    results_logger.setLevel(logging.INFO)
    results_logger.addHandler(logging.FileHandler('results/' + filename + '.log', 'w'))
    for epoch, epoch_result in results_per_epoch.items():
        log_results_per_epoch(results_logger, epoch, epoch_result)


if __name__ == "__main__":
    # load train&test data + relations dict
    relation_dict = extract_relation_dict(get_all_paths('training'))

    # hyperparams = ArcsTrainerHyperparameters(no_epochs=2,
    #                                          mlp_dropout=0.5,
    #                                          unaligned_tolerance=0)
    # run_experiment(relation_dict, hyperparams)

    hyperparams = ArcsTrainerHyperparameters(no_epochs=2,
                                             mlp_dropout=0.5,
                                             unaligned_tolerance=0,
                                             max_sen_len=300,
                                             max_parents_vectors=6,
                                             reentrancy_threshold=0.8,
                                             use_preprocessing=True,
                                             trainable_embeddings_size=128,
                                             glove_embeddings_size=100,
                                             lstm_out_dim=50,
                                             mlp_dim=32,
                                             no_lstm_layers=1)
    run_experiment(relation_dict, hyperparams)
