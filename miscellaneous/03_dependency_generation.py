from feature_extraction import dataset_loader

# training_data = dataset_loader.read_data("training")
dev_data = dataset_loader.read_data("dev")
# test_data = dataset_loader.read_data("test")

# data = training_data + dev_data + test_data
data = dev_data

dev_data_orig = dataset_loader.read_original_graphs("dev")

data_orig = dev_data_orig

print ""