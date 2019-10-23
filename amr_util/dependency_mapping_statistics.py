from data_extraction import dataset_loader
from feature_extraction import feature_vector_generator


def load_and_align_data():
    training_data = dataset_loader.read_data("training", "r2")
    dev_data = dataset_loader.read_data("dev", "r2")
    test_data = dataset_loader.read_data("test", "r2")

    data = training_data + dev_data + test_data

    processed_amr_ids = []
    filtered_data = []
    for i in range(len(data)):
        if data[i].amr_id not in processed_amr_ids:
            filtered_data.append(data[i])
            processed_amr_ids.append(data[i].amr_id)
    data = sorted(filtered_data, key=lambda d: d.amr_id)

    sequences, _, dependencies, _, amr_ids, _, _ = feature_vector_generator.extract_data_components(data)

    training_data_orig = dataset_loader.read_original_graphs("training", "r2")
    dev_data_orig = dataset_loader.read_original_graphs("dev", "r2")
    test_data_orig = dataset_loader.read_original_graphs("test", "r2")

    data_orig = training_data_orig + dev_data_orig + test_data_orig

    processed_orig_amr_ids = []
    filtered_data_orig = []
    for i in range(len(data_orig)):
        if data_orig[i][0] in amr_ids and data_orig[i][0] not in processed_orig_amr_ids:
            filtered_data_orig.append(data_orig[i])
            processed_orig_amr_ids.append(data_orig[i][0])
    data_orig = sorted(filtered_data_orig, key=lambda d: d[0])

    sentences = [d[1] for d in data_orig]
    amrs = [d[2] for d in data_orig]

    return sequences, dependencies, amrs, sentences


def get_direct_dependencies_pairs(dependencies_dict):
    head_dependant_tokens_pairs = []

    for dep_parent, (dep_child, dep_type) in dependencies_dict.items():
        head_dependant_tokens_pairs.append((dep_parent, dep_child))

    return head_dependant_tokens_pairs


def get_intermediary_step_dependencies_pairs(dependencies_dict):
    head_dependant_tokens_pairs = []

    for dep_parent, (dep_intermediary, _) in dependencies_dict.items():
        dep_intermediary_rel = dependencies_dict.get(dep_intermediary, None)
        if dep_intermediary_rel is not None and dep_intermediary_rel[1] != "ROOT":
            dep_child = dep_intermediary_rel[0]
            head_dependant_tokens_pairs.append((dep_parent, dep_child))

    return head_dependant_tokens_pairs


def get_two_intermediary_step_dependencies_pairs(dependencies_dict):
    head_dependant_tokens_pairs = []

    for dep_parent, (dep_first_intermediary, _) in dependencies_dict.items():
        dep_first_intermediary_rel = dependencies_dict.get(dep_first_intermediary, None)
        if dep_first_intermediary_rel is not None and dep_first_intermediary_rel[1] != "ROOT":
            dep_second_intermediary = dep_first_intermediary_rel[0]
            dep_second_intermediary_rel = dependencies_dict.get(dep_second_intermediary, None)
            if dep_second_intermediary_rel is not None and dep_second_intermediary_rel[1] != "ROOT":
                dep_child = dep_second_intermediary_rel[0]
                head_dependant_tokens_pairs.append((dep_parent, dep_child))

    return head_dependant_tokens_pairs


def get_amr_node_for_token(token, amr):
    for key, value in amr.node_to_tokens.items():
        if str(token) in value or str(token) in value[0]:
            return key
    return None


def amr_has_relation_for_nodes(node_1, node_2, amr):
    if node_1 and node_2:
        parent_node = node_1
        child_node = node_2

        if parent_node in amr:
            parent_rels = amr.get(parent_node)
            for rel in parent_rels.items():
                if tuple(child_node) in rel:
                    return "direct"

        parent_node = node_2
        child_node = node_1

        if parent_node in amr:
            parent_rels = amr.get(parent_node)
            for rel in parent_rels.items():
                if tuple(child_node) in rel:
                    return "inverse"

        return None

    else:
        return "unaligned"


if __name__ == "__main__":

    total_deps = 0
    direct_deps = 0
    inverse_deps = 0
    one_step_direct_deps = 0
    one_step_inverse_deps = 0
    two_step_direct_deps = 0
    two_step_inverse_deps = 0
    unaligned_node_deps = 0

    sequences, dependencies, amrs, sentences = load_and_align_data()

    for sequence, deps, amr, sentence in zip(sequences, dependencies, amrs, sentences):

        dep_pairs = get_direct_dependencies_pairs(deps)

        total_deps += len(dep_pairs)

        for dep_pair in dep_pairs:
            node_1 = get_amr_node_for_token(dep_pair[0], amr)
            node_2 = get_amr_node_for_token(dep_pair[1], amr)

            amr_rel = amr_has_relation_for_nodes(node_1, node_2, amr)

            if amr_rel is not None:
                if amr_rel == "direct":
                    direct_deps += 1
                elif amr_rel == "inverse":
                    inverse_deps += 1
                elif amr_rel == "unaligned":
                    unaligned_node_deps += 1

        dep_pairs = get_intermediary_step_dependencies_pairs(deps)

        for dep_pair in dep_pairs:
            node_1 = get_amr_node_for_token(dep_pair[0], amr)
            node_2 = get_amr_node_for_token(dep_pair[1], amr)

            amr_rel = amr_has_relation_for_nodes(node_1, node_2, amr)

            if amr_rel is not None:
                if amr_rel == "direct":
                    one_step_direct_deps += 1
                elif amr_rel == "inverse":
                    one_step_inverse_deps += 1
                elif amr_rel == "unaligned":
                    pass

        dep_pairs = get_two_intermediary_step_dependencies_pairs(deps)

        for dep_pair in dep_pairs:
            node_1 = get_amr_node_for_token(dep_pair[0], amr)
            node_2 = get_amr_node_for_token(dep_pair[1], amr)

            amr_rel = amr_has_relation_for_nodes(node_1, node_2, amr)

            if amr_rel is not None:
                if amr_rel == "direct":
                    two_step_direct_deps += 1
                elif amr_rel == "inverse":
                    two_step_inverse_deps += 1
                elif amr_rel == "unaligned":
                    pass

    present_deps = direct_deps + inverse_deps

    print "Data length: %d" % len(sequences)
    print "Total number of dependencies between tokens: %d" % total_deps
    print "Total number of dependencies between tokens that have an AMR relation: %d" % present_deps
    print "Percentage of mapped dependency relations: %f" % (float(present_deps) / total_deps)
    print "Percentage of direct dependencies: %f" % (float(direct_deps) / present_deps)
    print "Percentage of inverse dependencies: %f" % (float(inverse_deps) / present_deps)
    print "Unaligned node dependencies: %f" % (float(unaligned_node_deps) / total_deps)
    print "Number of AMR relations mapped through 2 consecutive dependency relations (direct and inverse): %d %d" % (
        one_step_direct_deps, one_step_inverse_deps)
    print "Number of AMR relations mapped through 3 consecutive dependency relations (direct and inverse): %d %d" % (
        two_step_direct_deps, two_step_inverse_deps)
