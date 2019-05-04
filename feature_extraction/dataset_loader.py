from os import listdir, path, makedirs
import math
import pickle as js
import numpy as np

from definitions import AMR_ALIGNMENTS_SPLIT
from models.amr_graph import AMR
from models.amr_data import CustomizedAMR
from preprocessing import SentenceAMRPairsExtractor
from training_data_extractor import generate_training_data


def partition_dataset(original_data_partitions, partition_sizes=None):
    """
        Partition a dataset into a set of partitions according to a list of sizes for each partition
    """
    if partition_sizes is not None:
        data = np.concatenate(original_data_partitions)
        data_len = len(data)

        partition_indices = [part_no + 1 for part_no in range(len(partition_sizes))]

        partition_boundaries = [int(math.floor(data_len * sum(partition_sizes[0:part_index])))
                                for part_index in partition_indices]
        partition_boundaries = [0] + partition_boundaries

        data_partitions = [data[partition_boundaries[part_idx - 1]: partition_boundaries[part_idx]]
                           for part_idx in partition_indices]

        return data_partitions
    else:
        return original_data_partitions


def read_original_graphs(type, filter_path="deft", cache=True):
    """
        Return a list of (amr_id, sentence, AMR, CustomizedAMR) quadruples
        Load the list from a dump file if present, else generate it and save it to a dump file
    """
    if filter_path is None:
        filter_path = "deft"
    dir_path = AMR_ALIGNMENTS_SPLIT + "/" + type
    print(dir_path + " with filter " + filter_path)

    parsed_data = []

    directory_content = listdir(dir_path)
    original_corpus = filter(lambda x: "dump" not in x and filter_path in x, directory_content)

    for file_name in original_corpus:
        original_file_path = dir_path + "/" + file_name
        dump_file_path = dir_path + "/original_graphs_dumps/" + file_name + ".dump"
        print(original_file_path)

        if cache and path.exists(dump_file_path):
            print("cache")
            with open(dump_file_path, "rb") as dump_file:
                parsed_data += js.load(dump_file)
        else:
            file_data = SentenceAMRPairsExtractor.extract_sentence_amr_pairs(original_file_path)

            parsed_file_data = []
            failed_amrs_in_file = 0

            for amr_triple in file_data:
                try:
                    camr_graph = AMR.parse_string(amr_triple[1])

                    custom_amr_graph = CustomizedAMR()
                    custom_amr_graph.create_custom_AMR(camr_graph)

                    parsed_file_data.append((amr_triple[2], amr_triple[0], camr_graph, custom_amr_graph))
                except Exception as _:
                    print "Exception when parsing AMR with ID: %s in file: %s\n" % (amr_triple[2], original_file_path)
                    failed_amrs_in_file += 1

            if not path.exists(path.dirname(dump_file_path)):
                makedirs(path.dirname(dump_file_path))
            with open(dump_file_path, "wb") as dump_file:
                js.dump(parsed_file_data, dump_file)
            parsed_data += parsed_file_data

            print "%d / %d in %s" % (failed_amrs_in_file, len(file_data), original_file_path)

    return parsed_data


def read_data(type, filter_path="deft", cache=True):
    """
        Return a list of TrainData instances
        Load the list from a dump file if present, else generate it and save it to a dump file
    """
    if filter_path is None:
        filter_path = "deft"
    dir_path = AMR_ALIGNMENTS_SPLIT + "/" + type
    print(dir_path + " with filter " + filter_path)

    parsed_data = []

    directory_content = listdir(dir_path)
    original_corpus = filter(lambda x: "dump" not in x and filter_path in x, directory_content)

    for file_name in original_corpus:
        original_file_path = dir_path + "/" + file_name
        dump_file_path = dir_path + "/dumps/" + file_name + ".dump"
        print(original_file_path)

        if cache and path.exists(dump_file_path):
            print("cache")
            with open(dump_file_path, "rb") as dump_file:
                parsed_data += js.load(dump_file)
        else:
            file_data = generate_training_data(original_file_path).data
            if not path.exists(path.dirname(dump_file_path)):
                makedirs(path.dirname(dump_file_path))
            with open(dump_file_path, "wb") as dump_file:
                js.dump(file_data, dump_file)
            parsed_data += file_data

    return parsed_data


def generate_parsed_data_files():
    """
        Initialize all the parsed data files (dumps) by regenerating them
    """
    read_original_graphs("training", cache=False)
    read_original_graphs("dev", cache=False)
    read_original_graphs("test", cache=False)

    read_data("training", cache=False)
    read_data("dev", cache=False)
    read_data("test", cache=False)
