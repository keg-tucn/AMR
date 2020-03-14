from os import listdir, path, makedirs
import math
import pickle as js
import numpy as np

from definitions import AMR_ALIGNMENTS_SPLIT
from models.amr_graph import AMR
from models.amr_data import CustomizedAMR
from models.parameters import ParserParameters
from data_extraction import input_file_parser, training_data_extractor


def read_data(file_type, filter_path="deft", parser_parameters=ParserParameters(), cache=True):
    """
        Returns a list of TrainData instances
        Loads the list from a dump file if present, else generates it and saves it to a dump file
        :param file_type - dataset partition (training, dev or test)
        :param filter_path - filtering criteria for data files
        :param parser_parameters - set of parser parameters and flags
        :param cache - allow to load from dump file if true, else calculate from original file and save new dump
    """
    if filter_path is None:
        filter_path = "deft"
    dir_path = AMR_ALIGNMENTS_SPLIT + "/" + file_type

    parsed_data = []

    directory_content = listdir(dir_path)
    original_corpus = sorted([x for x in directory_content if "dump" not in x and filter_path in x])

    for file_name in original_corpus:
        original_file_path = dir_path + "/" + file_name
        dump_file_path = dir_path + "/dumps/" + file_name + ".dump"
        print(original_file_path)

        if cache and path.exists(dump_file_path):
            print("cache")
            with open(dump_file_path, "rb") as dump_file:
                parsed_data += js.load(dump_file)
        else:
            print("generate")
            file_data = training_data_extractor.generate_training_data(original_file_path,
                                                                       parser_parameters=parser_parameters).data
            if not path.exists(path.dirname(dump_file_path)):
                makedirs(path.dirname(dump_file_path))
            with open(dump_file_path, "wb") as dump_file:
                js.dump(file_data, dump_file)
            parsed_data += file_data

    return parsed_data


def read_original_graphs(file_type, filter_path="deft", cache=True):
    """
        Returns a list of (amr_id, sentence, AMR, CustomizedAMR) quadruples
        Loads the list from a dump file if present, else generates it and saves it to a dump file
        :param file_type - data set partition (training, dev or test)
        :param filter_path - filtering criteria for data files
        :param cache - allow to load from dump file if true, else calculate from original file and save new dump
    """
    if filter_path is None:
        filter_path = "deft"
    dir_path = AMR_ALIGNMENTS_SPLIT + "/" + file_type

    parsed_data = []

    directory_content = listdir(dir_path)
    original_corpus = sorted([x for x in directory_content if "dump" not in x and filter_path in x])

    for file_name in original_corpus:
        original_file_path = dir_path + "/" + file_name
        dump_file_path = dir_path + "/original_graphs_dumps/" + file_name + ".dump"
        print(original_file_path)

        if cache and path.exists(dump_file_path):
            print("cache")
            with open(dump_file_path, "rb") as dump_file:
                parsed_data += js.load(dump_file)
        else:
            print("generate")
            file_data = input_file_parser.extract_data_records(original_file_path)

            parsed_file_data = []
            failed_amrs_in_file = 0

            for amr_triple in file_data:
                try:
                    camr_graph = AMR.parse_string(amr_triple[1])

                    custom_amr_graph = CustomizedAMR()
                    custom_amr_graph.create_custom_AMR(camr_graph)

                    parsed_file_data.append((amr_triple[2], amr_triple[0], camr_graph, custom_amr_graph))
                except Exception as _:
                    # print "Exception when parsing AMR with ID: %s in file %s with error: %s\n" % (
                    #    amr_triple[2], file_name, e)
                    failed_amrs_in_file += 1

            if not path.exists(path.dirname(dump_file_path)):
                makedirs(path.dirname(dump_file_path))
            with open(dump_file_path, "wb") as dump_file:
                js.dump(parsed_file_data, dump_file)
            parsed_data += parsed_file_data

            print(("%d / %d in %s" % (failed_amrs_in_file, len(file_data), original_file_path)))

    return parsed_data


def partition_dataset(original_data_partitions, partition_sizes=None, shuffle_data=True):
    """
        Partition a dataset into a set of partitions according to a list of sizes for each partition
        :param original_data_partitions - list of original dataset partitions
        :param partition_sizes - list of percentages of each result partition relative to the whole data size
        :param shuffle_data - specifies whether the data should be shuffled before being partitioned
    """
    if partition_sizes is not None:
        data = np.concatenate(original_data_partitions)
        data_len = len(data)

        partition_indices = [part_no + 1 for part_no in range(len(partition_sizes))]

        partition_boundaries = [int(math.floor(data_len * sum(partition_sizes[0:part_index])))
                                for part_index in partition_indices]
        partition_boundaries = [0] + partition_boundaries

        if shuffle_data:
            np.random.shuffle(data)

        data_partitions = [data[partition_boundaries[part_idx - 1]: partition_boundaries[part_idx]]
                           for part_idx in partition_indices]

        return data_partitions
    else:
        return original_data_partitions


def generate_parsed_graphs_files():
    """
        Initialize all the parsed data files (dumps) with the AMRGraphs and AMRData structures
    """
    read_original_graphs("training", cache=False)
    read_original_graphs("dev", cache=False)
    read_original_graphs("test", cache=False)


def generate_parsed_data_files(parser_parameters):
    """
        Initialize all the parsed data files (dumps) with the TrainData structures
    """
    read_data("training", parser_parameters=parser_parameters, cache=False)
    read_data("dev", parser_parameters=parser_parameters, cache=False)
    read_data("test", parser_parameters=parser_parameters, cache=False)


def remove_overlapping_instances(partition_1, partition_2, remove_from=1):
    if remove_from == 1:
        partition_2_ids = [d.amr_id for d in partition_2]

        partition_1 = [inst for inst in partition_1 if inst.amr_id not in partition_2_ids]

        return partition_1, partition_2

    elif remove_from == 2:
        partition_1_ids = [d.amr_id for d in partition_1]

        partition_2 = [inst for inst in partition_2 if inst.amr_id not in partition_1_ids]

        return partition_1, partition_2


def check_data_partitions_overlap(partition_1, partition_2):
    partition_2_ids = [d.amr_id for d in partition_2]

    overlap_count = 0

    for d in partition_1:
        if d.amr_id in partition_2_ids:
            overlap_count += 1

    return overlap_count
