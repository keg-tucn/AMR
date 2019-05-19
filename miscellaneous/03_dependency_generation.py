from data_extraction import dataset_loader
from data_extraction.dependency_extractor import DependencyExtractor

training_data = dataset_loader.read_data("training")
dev_data = dataset_loader.read_data("dev")
test_data = dataset_loader.read_data("test")

data = training_data + dev_data + test_data

training_data_orig = dataset_loader.read_original_graphs("training")
dev_data_orig = dataset_loader.read_original_graphs("dev")
test_data_orig = dataset_loader.read_original_graphs("test")

data_orig = training_data_orig + dev_data_orig + test_data_orig

for id, sentence, amr_graph, amr_data in data_orig:
    stanford_deps = DependencyExtractor.extract_dependencies(sentence)
    spaCy_deps = DependencyExtractor.extract_dependencies_spacy(sentence)
