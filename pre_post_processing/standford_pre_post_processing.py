from copy import deepcopy
from typing import List, Dict

from models.amr_graph import AMR
from models.concept import IdentifiedConcepts, Concept
from pre_post_processing.stanford_postprocessing_util import find_special_token_index, \
    create_subgraph_for_person_or_organization, add_subgraph, replace_special_token_with_original_node
from pre_post_processing.stanford_train_preprocessing_util import get_person_or_organization_tokens, \
    modify_sentence_and_alignment_for_person_or_organization, replace_subgraph_for_person_or_organization
from preprocessing.NamedEntitiesReplacer import process_sentence


def train_pre_processing(amr: AMR, sentence: str):
    """
    Preproceses the sentence and amr for person and organization on the train flow
        Takes as input a (amr, sentence) pair
        Returns a preprocessed (sentence,amr) pair
        Replaces the person/organization rooted subgraph in the AMR with the node PERSON/ORGANIZATION
        Replaces the Person/Organization NE in the sentence with the token PERSON/ORGANIZATION
        Creates a metadata dict (todo explain further)
    """
    # get variables of person and organization nodes
    amr_copy = deepcopy(amr)
    person_and_org_nodes_vars = []
    for var, concept in amr_copy.node_to_concepts.items():
        if concept == 'person' or concept == 'organization':
            children_dict = amr_copy[var]
            if 'name' in children_dict.keys() and 'wiki' in children_dict.keys():
                person_and_org_nodes_vars.append(var)
    new_sentence = sentence
    metadata_dict: Dict[int, List[str]] = {}
    for node_var in person_and_org_nodes_vars:
        # get tokens to be removed
        to_remove_tokens = get_person_or_organization_tokens(amr, sentence.split(), node_var)
        # if the tokens are not in the sentence, do not preprocess
        if len(to_remove_tokens) != 0:
            # remove from sentence & modify alignment
            new_sentence = modify_sentence_and_alignment_for_person_or_organization(amr_copy,
                                                                                    new_sentence,
                                                                                    node_var,
                                                                                    to_remove_tokens,
                                                                                    metadata_dict)
            # remove from graph
            replace_subgraph_for_person_or_organization(amr_copy, node_var)
    return amr_copy, new_sentence, metadata_dict


def post_processing_on_parent_vector(identified_concepts: IdentifiedConcepts,
                                     vector_of_parents: List[List[int]],
                                     preprocessed_sentence: str,
                                     preprocessing_metadata: Dict[int, List[str]]):
    """
    Applies post processing on list of concepts and vector of parents
    Adds the new concepts at the end of the list of concepts (ordered is no longer maintained,
    but should not be an issue at this point)
    Does the reconstruction based on the preprocessed sentence, concepts and metadata, trying to match
    PERSON or ORGANIZATION tokens from the sentence with concepts in the list and using the original tokens
    from the metadata
    """
    sentence_tokens = preprocessed_sentence.split()
    for i in range(len(sentence_tokens)):
        if sentence_tokens[i] == 'PERSON' or sentence_tokens[i] == 'ORGANIZATION':
            concept_index = find_special_token_index(identified_concepts, sentence_tokens[i])
            if concept_index != -1:
                # apply postprocessing
                original_tokens = preprocessing_metadata[i]
                subgraph_concepts, subgraph_parents = create_subgraph_for_person_or_organization(original_tokens)
                add_subgraph(identified_concepts, vector_of_parents, concept_index,
                             subgraph_concepts, subgraph_parents)
                replace_special_token_with_original_node(identified_concepts, concept_index)


def inference_preprocessing(sentence: str):
    """
    Takes a sentence and returns
        the preprocessed sentence (with NE replaced with PERSON or ORGANIZATION)
        a list with the initial tokens and where they have to be inserted
    """
    new_sentence, named_entities_locations = process_sentence(sentence, tags_to_identify=['PERSON', 'ORGANIZATION'])
    named_entities_location_dict = {}
    for named_entities_location in named_entities_locations:
        index, token_list = named_entities_location
        named_entities_location_dict[index] = token_list
    return new_sentence, named_entities_location_dict
