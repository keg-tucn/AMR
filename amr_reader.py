from os import listdir, path, makedirs
import pickle as js

from definitions import PROJECT_ROOT_DIR
from data_extraction import training_data_extractor as tde
from models.parameters import ParserParameters


def generate_parsed_data(parsed_path, cache, dump_path):
    dump_path = dump_path + ".dump"
    # print(dump_path)
    # don't cache it
    # if path.exists(dump_path):
    #    with open(dump_path, "rb") as f:
    #        return js.load(f)
    if cache:
        print("cache")
        if path.exists(dump_path):
            with open(dump_path, "rb") as f:
                return js.load(f)
    else:
        data = tde.generate_training_data(parsed_path, parser_parameters=ParserParameters()).data
        if not path.exists(path.dirname(dump_path)):
            makedirs(path.dirname(dump_path))
        with open(dump_path, "wb") as f:
            js.dump(data, f)  # , indent=4, separators=(',', ': ')
        return data


def read_data(type, cache, filter_path="deft"):
    if filter_path is None:
        filter_path = "deft"
    mypath = PROJECT_ROOT_DIR + '/resources/alignments/split/' + type
    print PROJECT_ROOT_DIR
    print(mypath + " with filter " + filter_path)
    data = []
    directory_content = listdir(mypath)
    original_corpus = filter(lambda x: "dump" not in x, directory_content)
    original_corpus = filter(lambda x: filter_path in x, original_corpus)
    for f in original_corpus:
        mypath_f = mypath + "/" + f
        dumppath_f = mypath + "/dumps/" + f
        print(mypath_f)
        data += generate_parsed_data(mypath_f, cache, dumppath_f)
    return data
