from os import listdir

from definitions import AMR_ALIGNMENTS_SPLIT, PROJECT_ROOT_DIR
from preprocessing import SentenceAMRPairsExtractor
from feature_extraction import dataset_loader

dataset_partitions = ["training", "dev", "test"]

dataset_versions = ["r1", "r2"]

default_path = AMR_ALIGNMENTS_SPLIT

with open(PROJECT_ROOT_DIR + "/resources/dataset_stats.txt", "w") as stats_file:
    for dataset_partition in dataset_partitions:
        partition_path = default_path + "/" + dataset_partition

        for dataset_version in dataset_versions:
            directory_content = listdir(partition_path)
            directory_content_filtered = filter(lambda x: dataset_version in x, directory_content)
            directory_content_filtered = sorted(directory_content_filtered)

            for file_name in directory_content_filtered:
                stats_file.write(file_name + "\n")
                file_path = partition_path + "/" + file_name
                stats_file.write(
                    "file length: %d\n" % len(SentenceAMRPairsExtractor.extract_sentence_amr_pairs(file_path)))
                stats_file.write(
                    "parsed AMRS: %d\n" % len(dataset_loader.read_original_graphs(dataset_partition, file_name)))
                stats_file.write("action seq. gen.: %d\n" % len(dataset_loader.read_data(dataset_partition, file_name)))
