from amr_util import frameset_matcher, nodebox_util, pos_converter


def annotate_node_concepts(node):
    node_label = node.label

    node_label = nodebox_util.simplify_word(node_label)

    children_tokens = [nodebox_util.simplify_word(n[0].label) for n in node.children]

    labels_with_source = []

    if nodebox_util.is_verb(node_label):
        labels_with_source.append((node_label, "propbank"))
    elif nodebox_util.is_noun(node_label):
        labels_with_source.append((node_label, "nombank"))
        labels_with_source.append((pos_converter.convert(node_label, "n", "v"), "propbank"))
    elif nodebox_util.is_adjective(node_label):
        labels_with_source.append((node_label, "nombank"))
        labels_with_source.append((pos_converter.convert(node_label, "a", "v"), "propbank"))
    elif nodebox_util.is_adverb(node_label):
        labels_with_source.append((node_label, "nombank"))
        labels_with_source.append((pos_converter.convert(node_label, "r", "v"), "propbank"))

    if len(labels_with_source) > 0:
        labels_with_sim = [frameset_matcher.compute_best_roleset(l[0], children_tokens, l[1]) for l in
                           labels_with_source]

        (frame, sim) = max(labels_with_sim, key=lambda l: l[1])
        if frame is not None:
            label_source = labels_with_source[labels_with_sim.index((frame, sim))][1]
            if label_source == "propbank":
                node.label = frame.id.replace(".", "-")
            else:
                node.label = frame.id.split(".")[0]
        else:
            node.label = node_label
    else:
        node.label = node_label

    for child in node.children:
        annotate_node_concepts(child[0])
