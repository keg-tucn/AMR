import copy
import re
from operator import itemgetter


def replace_named_entities(amr, sentence):
    amr_copy = copy.deepcopy(amr)
    # Find all the nodes which have a :name relation along with the node containing the "name" variable
    name_nodes = [(k, amr_copy[k]["name"][0]) for k in amr_copy if amr_copy[k] and "name" in amr_copy[k]]
    if len(name_nodes) == 0:
        return amr, sentence, []

    # Find the literals associated with each named entity
    literals_triplets = []
    for name_tuple in name_nodes:
        op_regexp = re.compile("^op([0-9])+$")
        name_var = name_tuple[1]
        op_rel_list = amr_copy[name_var]
        literals = []
        for op_rel in op_rel_list:
            if op_regexp.match(op_rel):
                literals.append(op_rel_list[op_rel][0])
        literals_triplets.append((name_tuple[0], name_tuple[1], literals))

    # Create a structure with named-entity-root-var, name-var, literals list, beginning of literal index, end of literal
    # index
    named_entities = []
    for literals_triplet in literals_triplets:
        literals_list = literals_triplet[2]
        tokens = [int(amr_copy.node_to_tokens[literal][0][0]) for literal in literals_list]
        named_entities.append((literals_triplet[0], literals_triplet[1], literals_triplet[2], min(tokens), max(tokens)))

    # Remove name vars from node_to_concepts
    name_variables = [n[1] for n in named_entities]
    amr_copy.node_to_concepts = dict((key, value) for key, value in amr_copy.node_to_concepts.iteritems()
                                     if key not in name_variables)

    # Remove literals from node_to_tokens
    literals = sum([n[2] for n in named_entities], [])
    amr_copy.node_to_tokens = dict((key, value) for key, value in amr_copy.node_to_tokens.iteritems()
                                   if key not in literals)

    # Remove name vars and literals from amr_copy dict
    for l in literals:
        if l in amr_copy.keys():
            amr_copy.pop(l)
    for n in name_variables:
        if n in amr_copy.keys():
            amr_copy.pop(n)

    # Update name root vars to have no children and remove wiki nodes
    name_roots = [n[0] for n in named_entities]
    for name_root in name_roots:
        if "wiki" in amr_copy[name_root].keys():
            if amr_copy[name_root]["wiki"][0] in amr_copy.keys():
                amr_copy.pop(amr_copy[name_root]["wiki"][0])
        amr_copy[name_root] = dict(
            (key, value) for key, value in amr_copy[name_root].iteritems() if key != "name" and key != "wiki")

    # Add name root vars in node_to_tokens and update incrementally the token indices of the affected nodes
    named_entities = sorted(named_entities, key=itemgetter(3))
    tokens = sentence.split(" ")
    total_displacement = 0
    for named_entity in named_entities:
        span_min = named_entity[3]
        span_max = named_entity[4]
        for n in amr_copy.node_to_tokens:
            amr_copy.node_to_tokens[n] = [t if isinstance(t, tuple) and int(t[0]) < span_max
                                          else (str(int(t[0]) - (span_max - span_min)), t[1]) if isinstance(t, tuple)
                                          else str(t) if int(t) < span_max
                                          else str(int(t) - (span_max - span_min))
                                          for t in amr_copy.node_to_tokens[n]]
        amr_copy.node_to_tokens[named_entity[0]] = [named_entity[3] - total_displacement]
        tokens = [tokens[:(span_min - total_displacement)] +
                  [amr_copy.node_to_concepts[named_entity[0]]] +
                  tokens[(span_max - total_displacement + 1):]][0]
        total_displacement = total_displacement + span_max - span_min
    sentence_copy = ' '.join(t for t in tokens)
    return amr_copy, sentence_copy, named_entities
