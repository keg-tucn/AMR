def get_unaligned_nodes(amr, unaligned_nodes_dict):
    for key in amr.keys():
        if key not in amr.node_to_tokens.keys():
            concept = key
            if key in amr.node_to_concepts.keys():
                concept = amr.node_to_concepts[key]
            if concept not in unaligned_nodes_dict.keys():
                unaligned_nodes_dict[concept] = [amr[key]]
            else:
                unaligned_nodes_dict[concept].append(amr[key])


def get_coreferences_count(custom_amr):
    keys = [k[0] for k in custom_amr.relations_dict.keys()]
    return len(keys) - len(set(keys))
