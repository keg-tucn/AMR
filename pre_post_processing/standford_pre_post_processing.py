from copy import deepcopy
from models.amr_graph import AMR
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
        Creates no metadata (has no associated train post processing)
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
    for node_var in person_and_org_nodes_vars:
        # get tokens to be removed
        to_remove_tokens = get_person_or_organization_tokens(amr, sentence.split(), node_var)
        # if the tokens are not in the sentence, do not preprocess
        if len(to_remove_tokens) != 0:
            # remove from sentence & modify alignment
            new_sentence = modify_sentence_and_alignment_for_person_or_organization(amr_copy,
                                                                                    new_sentence,
                                                                                    node_var,
                                                                                    to_remove_tokens)
            # remove from graph
            replace_subgraph_for_person_or_organization(amr_copy, node_var)
    return amr_copy, new_sentence


def inference_preprocessing(sentence: str):
    """
    Takes a sentence and returns
        the preprocessed sentence (with NE replaced with PERSON or ORGANIZATION)
        a list with the initial tokens and where they have to be inserted
    """
    new_sentence, named_entities_location = process_sentence(sentence, tags_to_identify=['PERSON', 'ORGANIZATION'])
    return new_sentence, named_entities_location
