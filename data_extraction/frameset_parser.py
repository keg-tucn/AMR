import os
import xml.etree.ElementTree as ET

from definitions import PROPBANK_FRAMES, NOMBANK_FRAMES
from models.frameset import Frameset


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
    if os.path.isfile(propbank_path):
        propbank_tree = ET.parse(propbank_path)
        propbank_instance = Frameset.build_from_XML(propbank_tree)

        return propbank_instance
    else:
        return None


def get_nombank_frameset(token):
    nombank_path = "%s/%s.xml" % (NOMBANK_FRAMES, token)
    if os.path.isfile(nombank_path):
        nombank_tree = ET.parse(nombank_path)
        nombank_instance = Frameset.build_from_XML(nombank_tree)

        return nombank_instance
    else:
        return None
