from os import listdir, path, makedirs
import pickle as js

from definitions import PROJECT_ROOT_DIR
from TrainingDataExtractor import generate_training_data


def read_data(type, filter_path="deft", cache=True):
    if filter_path is None:
        filter_path = "deft"
    dir_path = PROJECT_ROOT_DIR + '/resources/alignments/split/' + type
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
