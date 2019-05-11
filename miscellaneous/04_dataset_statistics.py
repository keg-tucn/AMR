from os import listdir

from definitions import AMR_ALIGNMENTS_SPLIT, PROJECT_ROOT_DIR
from preprocessing import SentenceAMRPairsExtractor
from feature_extraction import dataset_loader

dataset_partitions = ["training", "dev", "test"]

dataset_versions = ["r1", "r2"]

dataset_stats = {}

for dataset_partition in dataset_partitions:
    partition_path = AMR_ALIGNMENTS_SPLIT + "/" + dataset_partition
    for dataset_version in dataset_versions:
        directory_content = listdir(partition_path)
        directory_content_filtered = filter(lambda x: dataset_version in x, directory_content)
        directory_content_filtered = sorted(directory_content_filtered)
        for file_name in directory_content_filtered:
            file_path = partition_path + "/" + file_name
            dataset_name = file_name.split('-')[6].split('.')[0]
            file_length = len(SentenceAMRPairsExtractor.extract_sentence_amr_pairs(file_path))
            parsed_AMR = len(dataset_loader.read_original_graphs(dataset_partition, file_name))
            act_seq_gen = len(dataset_loader.read_data(dataset_partition, file_name))

            if dataset_name not in dataset_stats:
                dataset_stats[dataset_name] = {}
            if dataset_version not in dataset_stats[dataset_name]:
                dataset_stats[dataset_name][dataset_version] = {}

            dataset_stats[dataset_name][dataset_version][dataset_partition] = [file_length, parsed_AMR, act_seq_gen]

# small column width
scw = 7
# large column width
lcw = 12

with open(PROJECT_ROOT_DIR + "/resources/dataset_stats.txt", "w") as stats_file:
    stats_file.write("".rjust(lcw) + "R1".rjust(9 * scw) + "R2".rjust(9 * scw) + "\n")
    stats_file.write("".rjust(lcw) + ("train".rjust(3 * scw) + "dev".rjust(3 * scw) + "test".rjust(3 * scw)) * 2 + "\n")
    stats_file.write("File".rjust(lcw) + ("len".rjust(scw) + "AMR".rjust(scw) + "act".rjust(scw)) * 6)
    for dataset_name in sorted(dataset_stats.iterkeys()):
        stats_file.write("\n" + dataset_name.rjust(lcw))
        for dataset_version in dataset_versions:
            if dataset_version in dataset_stats[dataset_name]:
                for dataset_partition in dataset_partitions:
                    if dataset_partition in dataset_stats[dataset_name][dataset_version]:
                        stats = dataset_stats[dataset_name][dataset_version][dataset_partition]
                        stats_file.write(str(stats[0]).rjust(scw) + str(stats[1]).rjust(scw) + str(stats[2]).rjust(scw))
                    else:
                        stats_file.write("-".rjust(scw) * 3)
            else:
                stats_file.write("-".rjust(scw) * 9)
