import logging
import os
from copy import deepcopy

import dynet as dy

from data_extraction.dataset_reading_util import get_all_paths
from models.concept import Concept, IdentifiedConcepts
from models.node import Node
from pre_post_processing.standford_pre_post_processing import post_processing_on_parent_vector
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.arcs_trainer import test, ArcsDynetGraph, \
    load_arcs_model, predict_vector_of_parents, test_amr
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.arcs_trainer_util import \
    ArcsTrainerHyperparameters, generate_amr_node_for_vector_of_parents, calculate_smatch
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.training_arcs_data_extractor import \
    generate_arcs_training_data, ArcsTraingAndTestData, ArcsTrainingEntry
from trainers.arcs.head_selection.relations_dictionary_extractor import extract_relation_dict
from trainers.nodes.concept_extraction.sequence_to_sequence_ordered_concept_extraction.pointer_generator_concept_extractor.pointer_generator_trainer import \
    load_model as load_pointer_gen_concepts_model

from trainers.nodes.concept_extraction.sequence_to_sequence_ordered_concept_extraction.pointer_generator_concept_extractor.pointer_generator_trainer_util import \
    PointerGeneratorConceptExtractorGraphHyperparams, Cyclical_trainer
from trainers.nodes.concept_extraction.sequence_to_sequence_ordered_concept_extraction.training_concepts_data_extractor import \
    ConceptsTrainingEntry
from trainers.nodes.concept_extraction.sequence_to_sequence_ordered_concept_extraction.concept_extractor \
    import test as predict_concepts_from_sentence


def setup_loggers():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    overview_logger = logging.getLogger('overview_logs')
    overview_logger.setLevel(logging.INFO)
    overview_logger.addHandler(logging.FileHandler('logs/overview_logs.log', 'w'))

    detail_logger = logging.getLogger('detail_logs')
    detail_logger.setLevel(logging.INFO)
    detail_logger.addHandler(logging.FileHandler('logs/detail_logs.log', 'w'))
    return overview_logger, detail_logger


def setup_concept_log_files():
    if not os.path.exists('concept_logs'):
        os.makedirs('concept_logs')
    overview_logs = open('concept_logs/overview.txt', "w")
    detail_logs = open('concept_logs/detail', "w")
    return overview_logs, detail_logs


def close_concept_log_files(overview_logs, detail_logs):
    overview_logs.close()
    detail_logs.close()


def transform_arcs_to_concept_test_data(arcs_data: ArcsTraingAndTestData):
    new_entries = []
    for arcs_entry in arcs_data.test_entries:
        concept_entry = ConceptsTrainingEntry(arcs_entry.identified_concepts,
                                              arcs_entry.preprocessed_sentence,
                                              arcs_entry.logging_info,
                                              arcs_entry.amr_str)
        new_entries.append(concept_entry)
    return new_entries


def construct_ordered_concepts(concept_names):
    ordered_concepts = [Concept('', 'ROOT')]
    for i in range(0, len(concept_names)):
        concept_name = concept_names[i]
        # TODO: should predict is literal
        is_lit = False
        if is_lit:
            concept = Concept(concept_name, concept_name)
        else:
            concept = Concept('var', concept_name)
        ordered_concepts.append(concept)
    return ordered_concepts


if __name__ == "__main__":

    EXPERIMENTAL_RUN = False
    model = dy.Model()
    RUN_ON_20 = True
    if RUN_ON_20:
        max_sen_len = 20
        no_epochs_concepts = 30
    else:
        max_sen_len = 300
        no_epochs_concepts = 45
    alignment = 'isi'
    preprocessing = True
    concepts_hyperparams = PointerGeneratorConceptExtractorGraphHyperparams(no_epochs=no_epochs_concepts,
                                                                            max_sentence_len=max_sen_len,
                                                                            use_preprocessing=preprocessing,
                                                                            alignment=alignment,
                                                                            experimental_run=EXPERIMENTAL_RUN,
                                                                            two_classifiers=True,
                                                                            dropout=0.4,
                                                                            trainer=Cyclical_trainer)

    # load model
    arcs_hyperparams = ArcsTrainerHyperparameters(no_epochs=20,
                                                  mlp_dropout=0.5,
                                                  unaligned_tolerance=0,
                                                  max_sen_len=20,
                                                  max_parents_vectors=6,
                                                  reentrancy_threshold=0.8,
                                                  use_preprocessing=preprocessing,
                                                  trainable_embeddings_size=128,
                                                  glove_embeddings_size=100,
                                                  lstm_out_dim=50,
                                                  mlp_dim=32,
                                                  no_lstm_layers=1,
                                                  alignment=alignment,
                                                  experimental_run=EXPERIMENTAL_RUN)

    arcs_graph: ArcsDynetGraph = load_arcs_model(arcs_hyperparams, model)


    concepts_graph = load_pointer_gen_concepts_model(concepts_hyperparams, model)

    if EXPERIMENTAL_RUN:
        relation_dict = extract_relation_dict(list(get_all_paths('training')))
    else:
        train_dev_paths = list(get_all_paths('training')) + list(get_all_paths('dev'))
        relation_dict = extract_relation_dict(train_dev_paths)

    # read test data
    if EXPERIMENTAL_RUN:
        test_path = 'dev'
    else:
        test_path = 'test'
    test_entries, no_test_failed, no_pv_hist_test = generate_arcs_training_data(get_all_paths(test_path),
                                                                                arcs_hyperparams.unaligned_tolerance,
                                                                                arcs_hyperparams.max_sen_len,
                                                                                arcs_hyperparams.max_parents_vectors,
                                                                                arcs_hyperparams.use_preprocessing,
                                                                                False)
    train_test_data: ArcsTraingAndTestData = ArcsTraingAndTestData(train_entries=None,
                                                                   test_entries=test_entries,
                                                                   no_train_amrs=None,
                                                                   no_test_amrs=len(test_entries))

    i = 0
    avg_smatch = 0
    for arcs_test_entry in test_entries:
        sentence = arcs_test_entry.preprocessed_sentence
        sentence_tokens = sentence.split()
        predicted_concepts_names = concepts_graph.predict_sequence(sentence_tokens)
        if predicted_concepts_names[0] == 'ROOT':
            del predicted_concepts_names[0]
        gold_concept_names = [gold_concept.name for gold_concept in
                              arcs_test_entry.identified_concepts.ordered_concepts if gold_concept.name != 'ROOT']
        ordered_concepts = construct_ordered_concepts(predicted_concepts_names)

        predicted_vector_of_parents = predict_vector_of_parents(arcs_graph,
                                                                ordered_concepts,
                                                                arcs_hyperparams)
        predicted_vector_of_parents.insert(0, [-1])
        identified_concepts = IdentifiedConcepts()
        identified_concepts.ordered_concepts = ordered_concepts
        if len(ordered_concepts) == 1:
            print('Empty amr')
            predicted_amr_str = None
        else:
            if arcs_hyperparams.use_preprocessing:
                post_processing_on_parent_vector(identified_concepts,
                                                 predicted_vector_of_parents,
                                                 sentence,
                                                 arcs_test_entry.preprocessing_metadata)
            predicted_amr_node: Node = generate_amr_node_for_vector_of_parents(identified_concepts,
                                                                               predicted_vector_of_parents,
                                                                               relation_dict)
            predicted_amr_str = predicted_amr_node.amr_print_with_reentrancy()

            # calculate smatch
            gold_amr_str = arcs_test_entry.amr_str
            smatch_f_score = calculate_smatch(predicted_amr_str, gold_amr_str)
            avg_smatch += smatch_f_score

        if i % 100 == 0:
            print()
            print('sentence: ' + str(sentence_tokens))
            print('gold concepts: ' + str(arcs_test_entry.identified_concepts.ordered_concepts))
            print('gold concept names ' + str(gold_concept_names))
            print('predicted concepts: ' + str(predicted_concepts_names))
            if predicted_amr_str is not None:
                print('Predicted AMR:')
                print(predicted_amr_str)
                print('Gold amr')
                print(gold_amr_str)

    no_test_entries = len(test_entries)
    avg_smatch = avg_smatch / no_test_entries
    print('Avg smatch is: ' + str(avg_smatch))
    print('Done')
