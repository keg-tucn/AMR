from amr_util import frameset_matcher, nodebox_util, pos_converter


def annotate_node_concepts(node):
    node_label = node.label

    node_label = nodebox_util.simplify_word(node_label)

    children_tokens = [nodebox_util.simplify_word(n[0].label) for n in node.children]

    if nodebox_util.is_verb(node_label):
        node_label_verb = node_label
        node_label_noun = None
    elif nodebox_util.is_noun(node_label):
        node_label_noun = node_label
        node_label_verb = pos_converter.convert(node_label, "n", "v")
    elif nodebox_util.is_adjective(node_label):
        node_label_noun = node_label
        node_label_verb = pos_converter.convert(node_label, "a", "v")
    else:
        node_label_noun = node_label
        node_label_verb = None

    roleset_v, sim_v = frameset_matcher.compute_best_roleset(node_label_verb, children_tokens, "propbank")
    roleset_n, sim_n = frameset_matcher.compute_best_roleset(node_label_noun, children_tokens, "nombank")

    if roleset_v is not None and roleset_n is not None:
        if sim_v >= sim_n:
            roleset = roleset_v
        else:
            roleset = roleset_n
    else:
        if roleset_v is not None:
            roleset = roleset_v
        else:
            roleset = roleset_n

    if roleset is not None:
        if roleset == roleset_v:
            node.label = roleset.id.replace(".", "-")
        else:
            node.label = roleset.id.split(".")[0]

    for child in node.children:
        annotate_node_concepts(child[0])
