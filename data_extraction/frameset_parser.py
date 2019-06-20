from os import path, listdir
from tqdm import tqdm
import pickle as js
from xml.etree import ElementTree

from definitions import PROPBANK_FRAMES, PROPBANK_DUMP, NOMBANK_FRAMES, NOMBANK_DUMP
from models.frameset import Frameset


def extract_frames_to_dump_file(source):
    if source == "propbank":
        frames_dir = PROPBANK_FRAMES
        frames_dump_file = PROPBANK_DUMP
    elif source == "nombank":
        frames_dir = NOMBANK_FRAMES
        frames_dump_file = NOMBANK_DUMP
    else:
        frames_dir = ""
        frames_dump_file = ""

    frame_files = filter(lambda x: ".xml" in x, listdir(frames_dir))
    frames = {}

    for i in tqdm(range(len(frame_files))):
        file_path = frames_dir + "/" + frame_files[i]
        frame = Frameset.build_from_XML(ElementTree.parse(file_path))
        frames[frame.lemma] = frame

    with open(frames_dump_file, "wb") as dump_file:
        js.dump(frames, dump_file)


def load_frames(source):
    if source == "propbank":
        frames_dump_path = PROPBANK_DUMP
    elif source == "nombank":
        frames_dump_path = NOMBANK_DUMP
    else:
        frames_dump_path = ""

    if not path.isfile(frames_dump_path):
        extract_frames_to_dump_file(source)

    with open(frames_dump_path, "rb") as dump_file:
        return js.load(dump_file)


def get_merged_frameset(token):
    propbank = get_propbank_frameset(token)
    nombank = get_nombank_frameset(token)

    return Frameset.merge_framesets(propbank, nombank)


def get_frameset_from_bank(token, source):
    if source == "propbank":
        return get_propbank_frameset(token)
    if source == "nombank":
        return get_nombank_frameset(token)
    return None


def get_propbank_frameset(token):
    propbank_path = "%s/%s.xml" % (PROPBANK_FRAMES, token)
    if path.isfile(propbank_path):
        propbank_tree = ElementTree.parse(propbank_path)
        propbank_instance = Frameset.build_from_XML(propbank_tree)

        return propbank_instance
    else:
        return None


def get_nombank_frameset(token):
    nombank_path = "%s/%s.xml" % (NOMBANK_FRAMES, token)
    if path.isfile(nombank_path):
        nombank_tree = ElementTree.parse(nombank_path)
        nombank_instance = Frameset.build_from_XML(nombank_tree)

        return nombank_instance
    else:
        return None
