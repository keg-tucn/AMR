from feature_extraction import feature_vector_generator
from data_extraction import dataset_loader

# training_data = dataset_loader.read_data("training")
dev_data = dataset_loader.read_data("dev")
# test_data = dataset_loader.read_data("test")
# data = training_data + dev_data + test_data
data = dev_data

amrs = [d[2] for d in dataset_loader.read_original_graphs("dev")]

first_amr = amrs[0]

sequences, _, dependencies, _, _, _, _ = feature_vector_generator.extract_data_components(data)

for sequence, deps, amr, i in zip(sequences, dependencies, amrs, range(len(amrs))):
    # iterate over AMR relations dictionary in AMRGraph instance
    print "============ " + str(i) + " ============"
    for rel_key, rel_val in amr.iteritems():
        if rel_key in amr.node_to_tokens and isinstance(amr.node_to_tokens[rel_key][0], basestring):
            parent_index = int(amr.node_to_tokens[rel_key][0])
            for rel_type, rel_child in rel_val.iteritems():
                rel_child = rel_child[0][0]
                if rel_child in amr.node_to_tokens and isinstance(amr.node_to_tokens[rel_child][0], basestring):
                    child_index = int(amr.node_to_tokens[rel_child][0])
                    if parent_index in deps and deps[parent_index][0] is child_index:
                        # print str(feature_vector_generator.get_amr_rel_for_dep_rel(deps[parent_index][1])) + " | " + rel_type
                        print str(parent_index) + ": " + str(amr.node_to_concepts.get(rel_key)) + " | " + str(
                            child_index) + ": " + \
                              str(amr.node_to_concepts.get(rel_child)) + "| " + rel_type + " | " + str(
                            deps[parent_index][1])
