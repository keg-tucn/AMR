from typing import List

from models.concept import IdentifiedConcepts, Concept


def find_special_token_index(identified_concepts: IdentifiedConcepts, special_token: str):
    """
    Finds a special token (PERSON, ORGANIZATION) in the concept list and returns its index
    Returns -1 when concept not found
    """
    for i in range(len(identified_concepts.ordered_concepts)):
        concept = identified_concepts.ordered_concepts[i]
        if concept.name == special_token:
            return i
    return -1


def create_subgraph_for_person_or_organization(original_tokens: List[str]):
    """
    Creates a subgraph (list of concepts and vector of parents) for PERSON or ORGANIZATION
    Has as input the original tokens in the sentence
    """
    # concepts list
    concepts: List[Concept] = []
    wiki_literal = '_'.join(original_tokens)
    wiki_literal_node = Concept(wiki_literal, wiki_literal)
    name_node = Concept('','name')
    op_nodes = []
    for original_token in original_tokens:
        op_nodes.append(Concept(original_token,original_token))
    concepts.append(wiki_literal_node)
    concepts.append(name_node)
    concepts.extend(op_nodes)

    # parent vector
    vector_of_parents = []
    # wiki
    vector_of_parents.append([-1])
    # name
    vector_of_parents.append([-1])
    # op literals (have parent name node)
    op_literals_parent = [[1] for i in range(len(op_nodes))]
    vector_of_parents.extend(op_literals_parent)

    return concepts, vector_of_parents


def add_subgraph(identified_concepts: IdentifiedConcepts,
                 vector_of_parents: List[List[int]],
                 concept_index: int,
                 subgraph_concepts: List[Concept],
                 subgraph_parents: List[List[int]]):
    """
    Adds a subgraph given by concepts and vector of parents
    Modifies the graph identified_concepts and vector of parents
    """
    initial_no_graph_concepts = len(identified_concepts.ordered_concepts)

    # update list of concepts
    identified_concepts.ordered_concepts.extend(subgraph_concepts)

    # update parent vector
    new_subgraph_parents = []
    for parents in subgraph_parents:
        parent = parents[0]
        if parent == -1:
            new_parent = concept_index
        else:
            new_parent = parent + initial_no_graph_concepts
        new_subgraph_parents.append([new_parent])
    vector_of_parents.extend(new_subgraph_parents)


def replace_special_token_with_original_node(identified_concepts: IdentifiedConcepts,concept_index):
    concept = identified_concepts.ordered_concepts[concept_index]
    concept.name = concept.name.lower()