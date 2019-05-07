from feature_extraction import dataset_loader, feature_vector_generator

training_data = dataset_loader.read_data("training")
dev_data = dataset_loader.read_data("dev")
test_data = dataset_loader.read_data("test")

data = training_data + dev_data + test_data

sequences, _, dependencies, _, _, _, _ = feature_vector_generator.extract_data_components(data)

training_data_orig = dataset_loader.read_original_graphs("training")
dev_data_orig = dataset_loader.read_original_graphs("dev")
test_data_orig = dataset_loader.read_original_graphs("test")

data_orig = training_data_orig + dev_data_orig + test_data_orig

sentences = [d[1] for d in data_orig]
amrs = [d[2] for d in data_orig]

total_deps = 0
present_deps = 0

for sequence, deps, amr, sentence in zip(sequences, dependencies, amrs, sentences):

    total_deps += len(deps)

    for dep_parent, (dep_child, dep_type) in deps.iteritems():

        print str(dep_parent) + " -> " + dep_type + " -> " + str(dep_child)

        parent_node = None
        for key, value in amr.node_to_tokens.iteritems():
            if [str(dep_parent)] == value:
                parent_node = key

        child_node = None
        for key, value in amr.node_to_tokens.iteritems():
            if [str(dep_child)] == value:
                child_node = key

        if parent_node and child_node:
            if parent_node in amr:
                parent_rels = amr.get(parent_node)
                for rel in parent_rels.iteritems():
                    if tuple(child_node) in rel[1]:
                        present_deps += 1

            if child_node in amr:
                child_rels = amr.get(child_node)
                for rel in child_rels.iteritems():
                    if tuple(parent_node) in rel[1]:
                        present_deps += 1

print "Total number of dependencies between tokens: %d" % total_deps
print "Total number of dependencies between tokens that have an AMR relation: %d" % present_deps
print "Percentage of mapped dependency relations: %f" % (float(present_deps) / total_deps)
