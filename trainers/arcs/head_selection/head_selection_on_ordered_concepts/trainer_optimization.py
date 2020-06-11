import os
import logging
from data_extraction.dataset_reading_util import get_all_paths
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.trainer import train_and_test
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.trainer_util import ArcsTrainerHyperparameters, \
    plot_train_test_acc_loss, log_results_per_epoch

from trainers.arcs.head_selection.relations_dictionary_extractor import extract_relation_dict


def run_experiment(relation_dict, hyperparams: ArcsTrainerHyperparameters, filename=None):
    if filename is None:
        filename = str(hyperparams)
    results_per_epoch = train_and_test(relation_dict, hyperparams)

    if not os.path.exists('plots'):
        os.makedirs('plots')
    plot_train_test_acc_loss('plots/' + filename + '.png', results_per_epoch)

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

    hyperparams = ArcsTrainerHyperparameters(no_epochs=20,
                                             mlp_dropout=0.5,
                                             unaligned_tolerance=0,
                                             compare_gold=1,
                                             max_sen_len=300,
                                             max_parents_vectors=6,
                                             reentrancy_threshold=0.8,
                                             use_preprocessing=True)
    run_experiment(relation_dict, hyperparams)
