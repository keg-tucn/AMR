import numpy as np

from data_extraction import dataset_loader

train_data = np.asarray(dataset_loader.read_data("training", "proxy"), dtype=object)
test_data = np.asarray(dataset_loader.read_data("test", "proxy"), dtype=object)

[train_data, test_data] = dataset_loader.partition_dataset((train_data, test_data), partition_sizes=[0.9, 0.1])

print "Proxy train size before: %d" % len(train_data)
print "Proxy test size before: %d" % len(test_data)

print "Train and test overlap before: %d" % dataset_loader.check_data_partitions_overlap(train_data, test_data)
train_data, test_data = dataset_loader.remove_overlapping_instances(train_data, test_data)
print "Train and test overlap after: %d" % dataset_loader.check_data_partitions_overlap(train_data, test_data)

print "Proxy train after size: %d" % len(train_data)
print "Proxy test after size: %d" % len(test_data)
