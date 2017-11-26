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
        self.have_org_role_exceptions = have_org_role_exceptions