from constants import __DEP_AMR_REL_TABLE, __AMR_RELATIONS

from feature_extraction import dataset_loader

training_data = dataset_loader.read_data("training")
dev_data = dataset_loader.read_data("dev")
test_data = dataset_loader.read_data("test")

data = training_data + dev_data + test_data

dependencies = set()
for dat in data:
    for dep in dat.dependencies.values():
        dependencies.add(dep[1])

dependencies = set([d.split("_")[0] for d in dependencies])
dependencies = sorted(list(dependencies))

print "Dependencies from dataset: %d" % len(dependencies)
print dependencies

mapped_dependencies = sorted(list(set(__DEP_AMR_REL_TABLE.keys())))
mapped_relations = sorted(list(set(__DEP_AMR_REL_TABLE.values())))

print "Dependencies from mapping file: %d" % len(mapped_dependencies)
print mapped_dependencies

print "Unmapped dependency relations:"
for dep in dependencies:
    if dep not in mapped_dependencies:
        print dep

print "-----------------------------------------"

print "Unmapped AMR relations:"
for rel in mapped_relations:
    if rel not in __AMR_RELATIONS:
        print rel
