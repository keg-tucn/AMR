from AMRGraph import AMR
from AMRData import CustomizedAMR
import ActionSequenceGenerator
import NamedEntityReplacer


def pretty_print(amr):
    for k in amr.keys():
        print "Key: %s" % (k)
        list = amr[k]
        if len(list) == 0:
            print "Leaf"
        for rel in list:
            print "%s -> %s" % (rel, list[rel][0])
        print ""


def generate_action_sequence(custom_amr, sentence):
    return ActionSequenceGenerator.generate_action_sequence(custom_amr, sentence)


def generate_custom_amr(amr):
    print "\nMappings between node variables and their corresponding concepts.\n"
    print amr.node_to_concepts

    print "\nMappings between nodes and all the aligned tokens: If the nodes don't have" \
          "a variable (polarity, literals, quantities, interrogatives), they specify both the aligned tokens " \
          "and the parent in order to uniquely identify them\n"
    print amr.node_to_tokens

    print "\nMappings between relations and tokens. Uniquely identified by also specifying the parent of that relation.\n"
    # TODO: since the clean-up of parents which are not actual variables is done at the final, we might end up
    # having parents such as 9~e.15 for the relations. However, as I've seen so far, these kind of nodes are usually leaves
    # so hopefully we won't have this problem
    print amr.relation_to_tokens

    print "\nMappings from a node to each child, along with the relation between them.\n"
    pretty_print(amr)

    print "\nAll the nodes in the amr should appear here.\n"
    print amr.keys()

    print "\nCreating custom AMR.\n"
    custom_AMR = CustomizedAMR()
    custom_AMR.create_custom_AMR(amr)
    print "\nCustom AMR token to concepts dict\n"
    print custom_AMR.tokens_to_concepts_dict
    print "\nCustom AMR relations dict\n"
    print custom_AMR.relations_dict
    print "\nCustom AMR parent dict\n"
    print custom_AMR.parent_dict
    return custom_AMR


def generate_amr_with_literals(amr_string, sentence):
    amr = AMR.parse_string(amr_string)
    return NamedEntityReplacer.replace_named_entities(amr, sentence)
