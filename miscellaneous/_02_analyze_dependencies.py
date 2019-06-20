from feature_extraction import feature_vector_generator
from data_extraction import dataset_loader

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

total_deps = 0
present_deps = 0

direct_deps = 0
inverse_deps = 0
direct_int_deps = 0
inverse_int_deps = 0
unaligned_node_deps = 0

for sequence, deps, amr, sentence in zip(sequences, dependencies, amrs, sentences):

    total_deps += len(deps)

    for dep_parent, (dep_child, dep_type) in deps.iteritems():

        parent_node = None
        for key, value in amr.node_to_tokens.iteritems():
            if str(dep_parent) in value or str(dep_parent) in value[0]:
                parent_node = key

        child_node = None
        for key, value in amr.node_to_tokens.iteritems():
            if str(dep_child) in value or str(dep_child) in value[0]:
                child_node = key

        if parent_node and child_node:
            if parent_node in amr:
                parent_rels = amr.get(parent_node)
                for rel in parent_rels.iteritems():
                    if tuple(child_node) in rel[1]:
                        present_deps += 1
                        direct_deps += 1

            if child_node in amr:
                child_rels = amr.get(child_node)
                for rel in child_rels.iteritems():
                    if tuple(parent_node) in rel[1]:
                        present_deps += 1
                        inverse_deps += 1
                    else:
                        int_node = rel[1][0][0]

        else:
            unaligned_node_deps += 1

print "Data length: %d" % len(data)
print "Total number of dependencies between tokens: %d" % total_deps
print "Total number of dependencies between tokens that have an AMR relation: %d" % present_deps
print "Total number of direct dependencies between tokens that have an AMR relation: %d" % direct_deps
print "Total number of inverse dependencies between tokens that have an AMR relation: %d" % inverse_deps
print "Percentage of mapped dependency relations: %f" % (float(present_deps) / total_deps)
print "Percentage of direct dependencies: %f" % (float(direct_deps) / present_deps)
print "Percentage of inverse dependencies: %f" % (float(inverse_deps) / present_deps)
print "Unaligned node dependencies: %f" % (float(unaligned_node_deps) / total_deps)
