import re
from copy import deepcopy
from typing import List, Dict

from models.amr_graph import AMR


def update_relation_to_tokens(amr: AMR, node_var: str, relations_to_remove: List[str]):
    keys_to_remove = []
    for rel, token_parent_list in amr.relation_to_tokens.items():
        if rel in relations_to_remove:
            to_remove = []
            for token_parent in token_parent_list:
                parent = token_parent[1]
                if parent == node_var:
                    to_remove.append(token_parent)
            for token_parent_to_remove in to_remove:
                token_parent_list.remove(token_parent_to_remove)
            if len(amr.relation_to_tokens[rel]) == 0:
                keys_to_remove.append(rel)
    for key in keys_to_remove:
        del amr.relation_to_tokens[key]


def replace_subgraph_for_person_or_organization(amr: AMR, node_var):
    """
    Replace a subgraph for a person or organization. The wiki node and name subgraph are removed
    and person is replaced with PERSON, while organization is replaced with ORGANIZATION

    Eg.
        p / person
               :wiki "Deng_Xiaoping"
               :name (n / name :op1 "Deng"~e.1 :op2 "Xiaoping"~e.2)
               :ARG0-of (h / have-rel-role-91 :ARG2 (c / comrade~e.0)))
    With
        p / PERSON
               :ARG0-of (h / have-rel-role-91 :ARG2 (c / comrade~e.0)))
    """
    # replaced amr[node_var]['name'][0][0] with amr[node_var].get('name')[0][0]
    # because the first option sometimes gives wrong results
    wiki_literal = amr[node_var].get('wiki')[0][0]
    name_node = amr[node_var].get('name')[0][0]
    name_op_literals = [name_op_tuple[0][0] for name_op_tuple in amr[name_node].values()]

    # remove from relation_to_tokens
    update_relation_to_tokens(amr, node_var, ['wiki', 'name'])
    update_relation_to_tokens(amr, name_node, amr[name_node].keys())

    # remove from default dict
    del amr[node_var]['wiki']
    del amr[node_var]['name']
    del amr[name_node]
    if wiki_literal in amr.keys():
        # the condition is necessary in case there are two or more - wiki literals
        if wiki_literal == '-':
            # do not delete it if there is also polarity in the amr
            all_relations_list = [item.keys() for item in amr.values()]
            all_relations = [item for sublist in all_relations_list for item in sublist]
            if 'polarity' not in all_relations:
                del amr[wiki_literal]
        else:
            del amr[wiki_literal]
    for name_op_literal in name_op_literals:
        if name_op_literal in amr.keys():
            del amr[name_op_literal]

    # remove from node_to_concepts
    del amr.node_to_concepts[name_node]

    # remove from node_to_tokens
    nodes_to_remove: List = name_op_literals.copy()
    nodes_to_remove.append(name_node)
    nodes_to_remove.append(wiki_literal)
    for node in nodes_to_remove:
        if node in amr.node_to_tokens.keys():
            del amr.node_to_tokens[node]

    # replace person -> PERSON and organization -> ORGANIZATION
    amr.node_to_concepts[node_var] = amr.node_to_concepts[node_var].upper()
    return amr


def get_person_or_organization_tokens(amr: AMR, sentence_tokens: List[str], node_var: str):
    tokens = []
    children_of_person = amr[node_var]
    name_child_var = children_of_person.get('name')[0][0]
    for rel, name_ops in amr[name_child_var].items():
        token = name_ops[0][0]
        tokens.append(token)
    # I only want the tokens that exist in the sentence (see bolt12_10510_9581.4)
    tokens = [token for token in tokens if token in sentence_tokens]
    return tokens


def modify_node_to_tokens_alignment(amr: AMR, alignment_mapping: Dict[int, int]):
    node_to_tokens_copy = deepcopy(amr.node_to_tokens)
    # print(str(amr))
    for key, node_tokens_list in node_to_tokens_copy.items():
        amr.node_to_tokens[key] = []
        for node_token in node_tokens_list:
            if type(node_token) is tuple:
                token, parent = node_token
                new_token = str(alignment_mapping[int(token)])
                amr.node_to_tokens[key].append((new_token, parent))
            else:
                new_token = str(alignment_mapping[int(node_token)])
                amr.node_to_tokens[key].append(new_token)


def modify_relation_to_tokens_alignment(amr: AMR, alignment_mapping: Dict[int, int]):
    relation_to_tokens_copy = deepcopy(amr.relation_to_tokens)
    for rel, token_parent_list in relation_to_tokens_copy.items():
        amr.relation_to_tokens[rel]: List = []
        for token_parent in token_parent_list:
            token, parent = token_parent
            new_token = str(alignment_mapping[int(token)])
            amr.relation_to_tokens[rel].append((new_token, parent))


def construct_alignment_mapping(sen_len: int, no_tokens_removed: int, removal_indexes):
    alignment_mapping = {}
    for i in range(0, sen_len):
        alignment_mapping[i] = i
    for removal_index in removal_indexes:
        for i in range(0, sen_len):
            if i > removal_index:
                old_mapping = alignment_mapping[i]
                alignment_mapping[i] = old_mapping - no_tokens_removed + 1
    return alignment_mapping


def get_indices_of_sublist_in_list(main_list: List, sub_list: List):
    indices = []
    for i in range(0, len(main_list) - len(sub_list) + 1):
        temp_array = main_list[i:i + len(sub_list)]
        if temp_array == sub_list:
            indices.append(i)
    return indices


def modify_sentence_and_alignment_for_person_or_organization(amr: AMR, sentence: str,
                                                             node_var,
                                                             to_remove_tokens,
                                                             metadata_dict: Dict[int, List[str]]):
    sentence_tokens = sentence.split()
    n = len(sentence_tokens)
    k = len(to_remove_tokens)
    removal_indexes = [i for i in range(n - k + 1) if sentence_tokens[i:i + k] == to_remove_tokens]
    # create a mapping between old and new alignment
    alignment_mapping = construct_alignment_mapping(len(sentence_tokens), len(to_remove_tokens), removal_indexes)
    # modify alignment
    # metadata alignment
    old_metadata_dict = deepcopy(metadata_dict)
    for old_index, values in old_metadata_dict.items():
        new_index = alignment_mapping[old_index]
        if new_index != old_index:
            del metadata_dict[old_index]
            metadata_dict[new_index] = values
    # node_to_tokens
    modify_node_to_tokens_alignment(amr, alignment_mapping)
    # make sure the new PERSON/ORGANIZATION node is aligned
    if node_var not in amr.node_to_tokens.keys():
        amr.node_to_tokens[node_var] = []
        for removal_index in removal_indexes:
            amr.node_to_tokens[node_var].append(str(removal_index))
    # relation_to_tokens
    modify_relation_to_tokens_alignment(amr, alignment_mapping)
    # modify sentence
    new_token = amr.node_to_concepts[node_var].upper()
    # make sure all occurances of to_remove_token are removed
    new_sentence = ' '.join(sentence_tokens)
    substring_to_replace = ' '.join(to_remove_tokens)
    replacement_indexes = get_indices_of_sublist_in_list(sentence_tokens, to_remove_tokens)
    new_sentence = new_sentence.replace(substring_to_replace, new_token)
    # construct metadata
    for replacement_index in replacement_indexes:
        # need to use alignment_mapping in case the same token list occurs more then once
        metadata_dict[alignment_mapping[replacement_index]] = substring_to_replace.split()
    return new_sentence
