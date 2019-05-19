import xml.etree.ElementTree as ET

from definitions import PROPBANK_FRAMES, NOMBANK_FRAMES
from models.frameset import Frameset


def print_frameset(frameset):
    print frameset.lemma
    for roleset in frameset.rolesets:
        print roleset.id, roleset.name
        for role in roleset.roles:
            print "\t", role.index, role.description


frameset_word = "buy"

propbank_path = "%s/%s.xml" % (PROPBANK_FRAMES, frameset_word)
propbank_tree = ET.parse(propbank_path)
propbank_instance = Frameset.build_from_XML(propbank_tree)

nombank_path = "%s/%s.xml" % (NOMBANK_FRAMES, frameset_word)
nombank_tree = ET.parse(nombank_path)
nombank_instance = Frameset.build_from_XML(nombank_tree)

print "Propbank version"
print_frameset(propbank_instance)

print "Nombank version"
print_frameset(nombank_instance)

print "Merged version"
print_frameset(Frameset.merge_framesets(propbank_instance, nombank_instance))
