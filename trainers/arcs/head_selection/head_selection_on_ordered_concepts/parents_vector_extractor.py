from typing import List
import itertools
import random
from models.amr_graph import AMR
from models.concept import IdentifiedConcepts


def generate_parent_list_vector(amr: AMR, identified_concepts: IdentifiedConcepts):
    """
        Generate a vector of concept_idx -> list of parents
            where concept_idx is the idx of the concept in the
            identified_concepts.ordered_concepts list
    """
    if len(amr.roots) != 1:
        raise Exception('Multiple roots not handled')
    # Assume that constants and negatives/interrogatives can't be parents (the nodes with no vars)

    # create two dictionaries for concept var -> concept and concept name -> concept list (for nodes with/no var)
    var_to_concepts_dict = {}
    # the concepts in this list should be ordered by token (at least for 0 unalignment tolerance)
    concept_name_to_concept_list_dict = {}
    for concept in identified_concepts.ordered_concepts:
        if concept.variable in amr.node_to_concepts.keys():
            var_to_concepts_dict[concept.variable] = concept
        else:
            if concept.name not in concept_name_to_concept_list_dict.keys():
                concept_name_to_concept_list_dict[concept.name] = []
            concept_name_to_concept_list_dict[concept.name].append(concept)

    children_list_per_parent = extract_children_list_per_parent_from_amr_items(amr)

    # create two dictionaries for concept -> parent list, one for concepts with variables, and one for concepts without vars
    concept_to_parentlist_var_dict = create_concept_to_parentlist_var_dict(children_list_per_parent,
                                                                           var_to_concepts_dict)
    concept_to_parentlist_no_var_dict = create_concept_to_parentlist_no_var_dict(identified_concepts.amr_id,
                                                                                 amr,
                                                                                 children_list_per_parent,
                                                                                 concept_name_to_concept_list_dict,
                                                                                 var_to_concepts_dict)

    concept_to_parent_list = {}
    concept_to_parent_list.update(concept_to_parentlist_no_var_dict)
    concept_to_parent_list.update(concept_to_parentlist_var_dict)
    parents_vector = [None] * len(identified_concepts.ordered_concepts)
    index = 0
    concept_to_index = {}
    for concept in identified_concepts.ordered_concepts:
        concept_to_index[concept] = index
        index += 1
    for i in range(0, len(identified_concepts.ordered_concepts)):
        if i == 0:
            # for fake root: ROOT node
            parents_vector[i] = [-1]
        else:
            concept = identified_concepts.ordered_concepts[i]
            if concept not in concept_to_parent_list.keys():
                if concept.variable == amr.roots[0]:
                    # root
                    parents_vector[i] = [0]
                else:
                    print(identified_concepts.amr_id)
                    raise Exception('non root concept has no parent')
            else:
                if concept.variable == amr.roots[0]:
                    # root
                    if parents_vector[i] is None:
                        parents_vector[i] = []
                    parents_vector[i].append(0)
                for parent in concept_to_parent_list[concept]:
                    parent_idx = concept_to_index[parent]
                    if parents_vector[i] is None:
                        parents_vector[i] = []
                    parents_vector[i].append(parent_idx)
    return parents_vector


def check_parent_vector_validity(parent_vector: List[int]):
    # check validity
    # the parent should not be itself
    # (happens with amrs where concepts and vars are mistaken, eg. bolt-eng-DF-170-181103-8889109_0034.14)
    for i in range(len(parent_vector)):
        if i == parent_vector[i]:
            return False
    if 0 not in parent_vector:
        raise Exception('No root in parent vector')
    return True


def generate_parent_vectors(amr: AMR, identified_concepts: IdentifiedConcepts, max_no_parent_vectors: int):
    """
        Go from (AMR,identified concepts) -> parents vectors
        Return parents_vector: list of parent vectors
    """
    parent_list_vector = generate_parent_list_vector(amr, identified_concepts)

    # go through the vector and for the node containng the root remove the other nodes
    # for now do not deal with reentrancy for the root node???
    # maybe I can deal with reentrancies for the root node??
    for i in range(0, len(parent_list_vector)):
        if 0 in parent_list_vector[i]:
            parent_list_vector[i] = [0]

    parents_vector = list(itertools.product(*parent_list_vector))

    valid_parent_vectors = []
    for parent_vector in parents_vector:
        if check_parent_vector_validity(parent_vector):
            valid_parent_vectors.append(parent_vector)

    if len(valid_parent_vectors) == 0:
        return None
    return valid_parent_vectors[:max_no_parent_vectors]


def extract_children_list_per_parent_from_amr_items(amr: AMR):
    children_list_per_parent = {}
    for parent_var, children_dict in amr.items():
        children_list = []
        children_dict = amr[parent_var]
        for rel, children_for_rel in children_dict.items():
            for child_var_tuple in children_for_rel:
                # child_var_tuple example: ('l2',)
                child_var = child_var_tuple[0]
                children_list.append(child_var)
        children_list_per_parent[parent_var] = children_list
    return children_list_per_parent


def create_concept_to_parentlist_var_dict(children_list_per_parent, var_to_concepts_dict):
    concept_to_parentlist_var_dict = {}
    # create the concept_to_parentlist_var_dict from amr.items()
    for parent_var, children_list in children_list_per_parent.items():
        if parent_var in var_to_concepts_dict.keys():
            parent_concept = var_to_concepts_dict[parent_var]
            for child_var in children_list:
                if child_var in var_to_concepts_dict.keys():
                    child_concept = var_to_concepts_dict[child_var]
                    if child_concept not in concept_to_parentlist_var_dict.keys():
                        concept_to_parentlist_var_dict[child_concept] = []
                    concept_to_parentlist_var_dict[child_concept].append(parent_concept)
    return concept_to_parentlist_var_dict


def create_concept_to_parentlist_no_var_dict(amr_id: str,
                                             amr: AMR,
                                             children_list_per_parent,
                                             concept_name_to_concept_list_dict,
                                             var_to_concepts_dict):
    concept_to_parentlist_no_var_dict = {}
    # create the concept_to_parentlist_var_dict from amr.node_to_tokens() where possible
    # -- because there might be multiple such nodes and can't differentiate between them in amr.items()
    # first construct a var -> (concepts,parents) dictionary
    var_to_concepts_parents = {}
    for var, amr_node_to_tokens_entry in amr.node_to_tokens.items():
        if var in concept_name_to_concept_list_dict.keys():
            # token list could have the form  [('10', 's2'), ('21', 's3')] or
            # something like [('2', 'e2'), ('11', 'e2'), ('2', 'e'), ('11', 'e')]
            # I need to order the parents, then associate them with the nodes
            parent_token_dict = IdentifiedConcepts.get_parents_tokens_list_for_no_var_node(amr_node_to_tokens_entry)
            # order parents
            parents = order_parents_for_multiple_no_var_concepts(parent_token_dict)
            concepts = concept_name_to_concept_list_dict[var]
            # if there are some parents missing (eg. unaligned nodes, take them from amr.items())
            if len(parents) < len(concepts):
                # take the parents
                parents = augment_with_parents_from_amr_items(parents, var, children_list_per_parent)
            if len(parents) > len(concepts):
                raise Exception('Constants should not be part of reentrancy ')
            # now parens and concepts should have the same length => associate them
            # concepts_parents = zip(concepts, parents)
            var_to_concepts_parents[var] = (concepts, parents)
            # for concept, parent_var in concepts_parents:
            #     concept_to_parentlist_no_var_dict[concept] = [var_to_concepts_dict[parent_var]]
    # make sure all the non vars are in the dict
    for var in concept_name_to_concept_list_dict.keys():
        if var!='ROOT' and var not in var_to_concepts_parents.keys():
            concepts = concept_name_to_concept_list_dict[var]
            parents = []
            var_to_concepts_parents[var] = (concepts, parents)
    for var, concepts_parents in var_to_concepts_parents.items():
        concepts = concepts_parents[0]
        parents = concepts_parents[1]
        if len(parents) < len(concepts):
            # take the parents
            parents = augment_with_parents_from_amr_items(parents, var, children_list_per_parent)
        if len(parents) > len(concepts):
            raise Exception('Constants should not be part of reentrancy ')
        concepts_parents_zipped = zip(concepts,parents)
        for concept, parent_var in concepts_parents_zipped:
            concept_to_parentlist_no_var_dict[concept] = [var_to_concepts_dict[parent_var]]
    return concept_to_parentlist_no_var_dict


def order_parents_for_multiple_no_var_concepts(parent_token_dict):
    """
    Input: smth like  {'e2':['2','7'],'e':['2','11'],'m':['9']}
    In this case it would return ['e2','m','e']
    """
    # transform it to have parent -> tokens set
    for parent, token_list in parent_token_dict.items():
        parent_token_dict[parent] = set(token_list)
    all_tokens_set = parent_token_dict.values()
    set_intersection = set.intersection(*all_tokens_set)
    # remove common tokens
    for parent, token_set in parent_token_dict.items():
        parent_token_dict[parent] = token_set.difference(set_intersection)
    parents = list(parent_token_dict.keys())

    def sort_by_token(parent):
        if not parent_token_dict[parent]:
            return -1
        # if token list not empty, compare by first token
        return int(list(parent_token_dict[parent])[0])

    parents.sort(key=sort_by_token)

    return parents


def augment_with_parents_from_amr_items(parents, var, children_list_per_parent):
    """
    For no variable nodes, augment with parents from amr.items when they couldn't
    be extracted from amr.node_to_tokens
    Input: list of ordered parents
    Output: list of ordered parents completed with some parents from amr.items() -- put randomly
    """
    parents_set = set(parents)
    all_parents = set()
    for parent, children_list in children_list_per_parent.items():
        for child in children_list:
            if child == var:
                all_parents.add(parent)
    parents_to_add = all_parents.difference(parents_set)
    for parent_to_add in parents_to_add:
        if len(parents) == 0:
            # simply add it
            parents.append(parent_to_add)
        else:
            # add it randomly in the parents list
            random_index = random.randint(0, len(parents) - 1)
            parents.insert(random_index, parent_to_add)
    return parents
