import re
from os import listdir

from data_extraction import input_file_parser
from definitions import AMR_ALIGNMENTS_SPLIT, JAMR_ALIGNMENTS_SPLIT, MERGED_ALIGNMENTS_SPLIT
from models.amr_data import CustomizedAMR
from models.amr_graph import AMR


def get_all_paths_for_alignment(split: str, alignment: str):
    if alignment == "jamr":
        dir_path = JAMR_ALIGNMENTS_SPLIT + "/" + split
    elif alignment == "merged":
        dir_path = MERGED_ALIGNMENTS_SPLIT + "/" + split
    else:
        dir_path = AMR_ALIGNMENTS_SPLIT + "/" + split

    directory_content = listdir(dir_path)
    original_corpus = sorted([x for x in directory_content if "dump" not in x])
    all_paths = []
    for file_name in original_corpus:
        original_file_path = dir_path + "/" + file_name
        print(original_file_path)
        all_paths.append(original_file_path)
    return all_paths


def get_all_paths(split: str):
    dir_path = AMR_ALIGNMENTS_SPLIT + "/" + split
    directory_content = listdir(dir_path)
    original_corpus = sorted([x for x in directory_content if "dump" not in x])
    all_paths = []
    for file_name in original_corpus:
        original_file_path = dir_path + "/" + file_name
        print(original_file_path)
        all_paths.append(original_file_path)
    return all_paths


def read_dataset_dict(split):
    """
    Returns the data by dataset for the given split(training, dev or test):
        Something like:
        {
            bolt: [(sentence, amr_str, amr_id)],
            deft-a: [(sentence, amr_str, amr_id)],
            etc.
        }
    """
    dataset_dict = {}
    paths = get_all_paths(split)
    for path in paths:
        result = re.search(split + '-(.*).txt', path)
        dataset = result.group(1)
        data = input_file_parser.extract_data_records(path)
        dataset_dict[dataset] = data
    return dataset_dict


