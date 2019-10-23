from semantic_relations_learner import concepts_relations_extractor

CONCEPTS_RELATIONS_DICT = concepts_relations_extractor.get_concepts_relations_pairs()


def annotate_node_relations(node):
    node_label = node.label

    if node.children is not None and len(node.children) > 0:
        node.children = [(node_child[0],
                          max(CONCEPTS_RELATIONS_DICT.get((node_label, node_child[0].label), [("unk", 1)]),
                              key=(lambda rel: rel[1]))[0])
                         if eligible_for_replace(node_label, node_child[0], node_child[1])
                         else (node_child[0], node_child[1])
                         for node_child in node.children]

    for child in node.children:
        annotate_node_relations(child[0])


def eligible_for_replace(parent_label, child_label, relation):
    return relation != "name" and relation != "wiki" and "op" not in relation and parent_label != "date-entity"
