import logging
import os
import pickle
from copy import deepcopy

import dynet as dy

from data_extraction.dataset_reading_util import get_all_paths, get_all_paths_for_alignment
from models.amr_graph import AMR
from models.concept import Concept, IdentifiedConcepts
from models.node import Node
from pre_post_processing.standford_pre_post_processing import post_processing_on_parent_vector, inference_preprocessing
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.arcs_trainer import test, ArcsDynetGraph, \
    load_arcs_model, predict_vector_of_parents, test_amr
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.arcs_trainer_util import \
    ArcsTrainerHyperparameters, generate_amr_node_for_vector_of_parents, calculate_smatch
from trainers.arcs.head_selection.head_selection_on_ordered_concepts.baseline import MLP_DROPOUT, MAX_PARENT_VECTORS, \
    REENTRANCY_THRESHOLD, PREPROCESSING, TRAINABLE_EMB_SIZE, GLOVE_EMB_SIZE, LSTM_OUT_DIM, MLP_DIM, NO_LSTM_LAYERS, \
    CHAR_CNN_CUTOFF, USE_VERB_FLAG, TRAINER
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


def construct_ordered_concepts(concept_names):
    ordered_concepts = [Concept('', 'ROOT')]
    for i in range(0, len(concept_names)):
        concept_name = concept_names[i]
        # TODO: should predict is literal
        is_lit = concept_name in ['-', 'imperative', 'expressive']
        is_lit = is_lit or (str(concept_name).startswith("\"") and str(concept_name).endswith("\""))
        if is_lit:
            concept = Concept(concept_name, concept_name)
        else:
            concept = Concept('var', concept_name)
        ordered_concepts.append(concept)
    return ordered_concepts


def get_amr_str(arcs_graph, arcs_hyperparams, ordered_concepts, relation_dict, sentence, metadata):
    predicted_vector_of_parents = predict_vector_of_parents(arcs_graph,
                                                            ordered_concepts,
                                                            arcs_hyperparams)
    predicted_vector_of_parents.insert(0, [-1])
    identified_concepts = IdentifiedConcepts()
    identified_concepts.ordered_concepts = ordered_concepts
    if len(ordered_concepts) == 1:
        print('Empty amr')
        return None
    else:
        if arcs_hyperparams.use_preprocessing:
            post_processing_on_parent_vector(identified_concepts,
                                             predicted_vector_of_parents,
                                             sentence,
                                             metadata)
        predicted_amr_node: Node = generate_amr_node_for_vector_of_parents(identified_concepts,
                                                                           predicted_vector_of_parents,
                                                                           relation_dict)
        return predicted_amr_node.amr_print_with_reentrancy()


def get_relation_dict(exp_run, alignment):
    if not os.path.exists('saved_rel_dir'):
        os.makedirs('saved_rel_dir')
    save_path = 'saved_rel_dir/rel_for_run_exp' + str(exp_run)
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            relation_dict = pickle.load(f)
            return relation_dict
    if exp_run:
        relation_dict = extract_relation_dict(list(get_all_paths_for_alignment('training', alignment)))
    else:
        train_dev_paths = list(get_all_paths_for_alignment('training', alignment)) + \
                          list(get_all_paths_for_alignment('dev', alignment))
        relation_dict = extract_relation_dict(train_dev_paths)

    with open(save_path, "wb") as f:
        print('Saving relation dict...')
        pickle.dump(relation_dict, f)

    return relation_dict

def predict_amr_for_sentence(sentence, prep = True):
    sentence_tokens = sentence.split()
    predicted_concepts_names = concepts_graph.predict_sequence(sentence_tokens)
    if predicted_concepts_names[0] == 'ROOT':
        del predicted_concepts_names[0]
    ordered_concepts = construct_ordered_concepts(predicted_concepts_names)
    if prep:
        preprocessed_sentence, metadata = inference_preprocessing(sentence)
    else:
        preprocessed_sentence = sentence
        metadata = []
    predicted_amr_str = get_amr_str(arcs_graph, arcs_hyperparams, ordered_concepts, relation_dict,
                                    preprocessed_sentence, metadata)
    return predicted_concepts_names, predicted_amr_str


def setup():
    LABELLED = True
    EXPERIMENTAL_RUN = False
    model = dy.Model()
    RUN_ON_20 = False
    RUN_ON_ORDERED = True
    GOLD_PREP = True
    if RUN_ON_20:
        max_sen_len = 20
        no_epochs_concepts = 30
    else:
        max_sen_len = 300
        no_epochs_concepts = 45
    alignment = 'jamr'
    preprocessing = True
    if RUN_ON_ORDERED:
        test_unatol = 0
    else:
        test_unatol = 1
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
                                                  mlp_dropout=MLP_DROPOUT,
                                                  unaligned_tolerance=0,
                                                  max_sen_len=max_sen_len,
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

    arcs_graph: ArcsDynetGraph = load_arcs_model(arcs_hyperparams, model)
    if arcs_graph is None:
        exit()

    concepts_graph = load_pointer_gen_concepts_model(concepts_hyperparams, model)

    if LABELLED:

        relation_dict = get_relation_dict(EXPERIMENTAL_RUN, alignment)

    else:
        relation_dict = {}
    return concepts_graph, arcs_graph, arcs_hyperparams, relation_dict


if __name__ == "__main__":
    concepts_graph, arcs_graph, arcs_hyperparams, relation_dict = setup()
    print('John likes cats')
    _,predicted_amr = predict_amr_for_sentence('John likes cats')
    print('Predicted amr: ')
    print(predicted_amr)
    print('John Smith likes cats')
    _,predicted_amr = predict_amr_for_sentence('John Smith likes cats')
    print('Predicted amr: ')
    print(predicted_amr)
    print('John Smith is the president')
    _,predicted_amr = predict_amr_for_sentence('John Smith is the president')
    print('Predicted amr: ')
    print(predicted_amr)

