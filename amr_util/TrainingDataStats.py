def get_unaligned_nodes(amr, unaligned_nodes_dict):
    for key in list(amr.keys()):
        if key not in list(amr.node_to_tokens.keys()):
            concept = key
            if key in list(amr.node_to_concepts.keys()):
                concept = amr.node_to_concepts[key]
            if concept not in list(unaligned_nodes_dict.keys()):
                unaligned_nodes_dict[concept] = [amr[key]]
            else:
                unaligned_nodes_dict[concept].append(amr[key])


def get_coreference_count(custom_amr):
    # TODO: fix method
    # but in case of reentrancy, a variable will have more than one parent (key here is a node-parent pair)
    keys = [k[0] for k in list(custom_amr.relations_dict.keys())]
    return len(keys) - len(set(keys))


class TrainingDataStatistics:

    def __init__(self, unaligned_nodes, unaligned_nodes_after, coreferences_count, named_entity_exceptions,
                 date_entity_exceptions, temporal_quantity_exceptions, quantity_exceptions, have_org_role_exceptions):
        self.unaligned_nodes = unaligned_nodes
        self.unaligned_nodes_after = unaligned_nodes_after
        self.coreferences_count = coreferences_count
        self.named_entity_exceptions = named_entity_exceptions
        self.date_entity_exceptions = date_entity_exceptions
        self.temporal_quantity_exceptions = temporal_quantity_exceptions
        self.quantity_exceptions = quantity_exceptions
        self.have_org_role_exlceptions = have_org_role_exceptions
