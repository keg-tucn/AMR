import copy
import re
from operator import itemgetter

from models.node import Node


def replace_date_entities(amr, sentence):
    amr_copy = copy.deepcopy(amr)

    date_rels = ["calendar",
                 "century",
                 "day",
                 "dayperiod",
                 "decade",
                 "month",
                 "quant",
                 "quarter",
                 "season",
                 "time",
                 "time-of",
                 "timezone",
                 "unit",
                 "weekday",
                 "year"]
    date_entities = [k for k in amr_copy.keys() if k in amr_copy.node_to_concepts.keys() and
                     amr_copy.node_to_concepts[k] == "date-entity"]

    if len(date_entities) == 0:
        return amr, sentence, []

    date_tuples = []
    for date_entity in date_entities:
        op_rel_list = amr_copy[date_entity]
        literals = []
        relations = []
        node = Node("date-entity")
        for op_rel in op_rel_list:
            if op_rel in date_rels:
                child = op_rel_list[op_rel][0]
                # if it"s not in node_to_tokens, all good
                if child not in amr_copy.node_to_concepts.keys():
                    literals.append(child)
                    relations.append(op_rel)
                    node.add_child(Node(None, child), op_rel)
        date_tuples.append((date_entity, literals, relations, node))

    date_entities = []
    for date_tuple in date_tuples:
        literals_list = date_tuple[1]
        tokens = [int(amr_copy.node_to_tokens[literal][0][0]) for literal in literals_list]
        date_entities.append((date_tuple[0], date_tuple[1], date_tuple[2], min(tokens), max(tokens), date_tuple[3]))

    # Remove literals from node_to_tokens
    literals = sum([d[1] for d in date_entities], [])
    amr_copy.node_to_tokens = dict((key, value) for key, value in amr_copy.node_to_tokens.iteritems()
                                   if key not in literals)

    for l in literals:
        if l in amr_copy.keys():
            amr_copy.pop(l)

    for date_entity in date_entities:
        amr_copy[date_entity[0]] = dict(
            (key, value) for key, value in amr_copy[date_entity[0]].iteritems() if key not in date_entity[2])

    # Add name root vars in node_to_tokens and update incrementally the token indices of the affected nodes
    date_entities = sorted(date_entities, key=itemgetter(3))
    tokens = sentence.split(" ")
    total_displacement = 0
    for date_entity in date_entities:
        span_min = date_entity[3]
        span_max = date_entity[4]
        for n in amr_copy.node_to_tokens:
            amr_copy.node_to_tokens[n] = [t if isinstance(t, tuple) and int(t[0]) < span_max
                                          else (str(int(t[0]) - (span_max - span_min)), t[1]) if isinstance(t, tuple)
            else str(t) if int(t) < span_max
            else str(int(t) - (span_max - span_min))
                                          for t in amr_copy.node_to_tokens[n]]
        amr_copy.node_to_tokens[date_entity[0]] = [date_entity[3] - total_displacement]
        tokens = [tokens[:(span_min - total_displacement)] +
                  [amr_copy.node_to_concepts[date_entity[0]]] +
                  tokens[(span_max - total_displacement + 1):]][0]
        total_displacement = total_displacement + span_max - span_min
    sentence_copy = " ".join(t for t in tokens)
    return amr_copy, sentence_copy, date_entities


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
        node = Node("name")
        for op_rel in op_rel_list:
            if op_regexp.match(op_rel):
                literal = op_rel_list[op_rel][0]
                literals.append(literal)
                literal_node = Node(None, "\"" + literal + "\"")
                node.add_child(literal_node, op_rel)
        root = Node(amr_copy.node_to_concepts[name_tuple[0]])
        root.add_child(node, "name")
        literals_triplets.append((name_tuple[0], name_tuple[1], literals, root))

    # Create a structure with named-entity-root-var, name-var, literals list, beginning of literal index, end of literal
    # index
    named_entities = []
    for literals_triplet in literals_triplets:
        literals_list = literals_triplet[2]
        tokens = [int(amr_copy.node_to_tokens[literal][0][0]) for literal in literals_list]
        named_entities.append((literals_triplet[0], literals_triplet[1], literals_triplet[2], min(tokens), max(tokens),
                               literals_triplet[3]))

    # Remove name vars from node_to_concepts
    name_variables = [n[1] for n in named_entities]
    amr_copy.node_to_concepts = dict((key, value) for key, value in amr_copy.node_to_concepts.iteritems()
                                     if key not in name_variables)

    # Remove literals from node_to_tokens
    literals = sum([n[2] for n in named_entities], [])
    amr_copy.node_to_tokens = dict((key, value) for key, value in amr_copy.node_to_tokens.iteritems()
                                   if key not in literals)

    # Remove name vars and literals from amr_copy_copy dict
    for l in literals:
        if l in amr_copy.keys():
            amr_copy.pop(l)
    for n in name_variables:
        if n in amr_copy.keys():
            amr_copy.pop(n)

    # Update name root vars to have no name and wiki children
    for name_entity in named_entities:
        name_root = name_entity[0]
        if "wiki" in amr_copy[name_root].keys():
            wiki_content = amr_copy[name_root]["wiki"][0]
            if wiki_content in amr_copy.keys():
                amr_copy.pop(wiki_content)
            name_entity[5].add_child(Node(None, "\"" + wiki_content + "\""), "wiki")
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
    sentence_copy = " ".join(t for t in tokens)
    return amr_copy, sentence_copy, named_entities


def replace_temporal_quantities(amr, sentence):
    amr_copy = copy.deepcopy(amr)

    # Find all the "temporal-quantity" nodes.
    temporal_quantity_nodes = [k for k in amr_copy if k in amr_copy.node_to_concepts
                               and amr_copy.node_to_concepts[k] == "temporal-quantity"]

    # Find the "quant" and "unit" nodes corresponding to the temporal quantity.
    quant_unit_tokens = [(k, amr_copy[k]["quant"][0], amr_copy[k]["unit"][0])
                         if "quant" in amr_copy[k] and "unit" in amr_copy[k]
                         else (k, "", "")
                         for k in temporal_quantity_nodes]

    quant_unit_tokens_align = [(t[0], t[1], t[2],
                                int(amr_copy.node_to_tokens[t[1]][0][0]),
                                int(amr_copy.node_to_tokens[t[2]][0]))
                               if t[1] in amr_copy.node_to_tokens and
                                  t[2] in amr_copy.node_to_tokens
                               else (t, "", "", -1, -1)
                               for t in quant_unit_tokens]

    for t in quant_unit_tokens_align:
        if t[3] == -1 or t[4] == -1:
            raise ValueError("Error! Unaligned quantity or token for sentence %s" % sentence)

        else:
            if not (abs(t[3] - t[4]) == 1 or
                    (abs(t[3] - t[4]) == 2
                     and sentence.split(" ")[max(t[3], t[4]) - 1] == "@-@")):
                raise ValueError("Quant and unit not consecutive or separated by @-@ for sentence %s" % sentence)
    # Remove units from node_to_concepts
    units = [t[2] for t in quant_unit_tokens_align]
    amr_copy.node_to_concepts = dict((key, value) for key, value in amr_copy.node_to_concepts.iteritems()
                                     if key not in units)

    # Remove quant and unit from node_to_tokens
    quants = [t[1] for t in quant_unit_tokens_align]
    amr_copy.node_to_tokens = dict((key, value) for key, value in amr_copy.node_to_tokens.iteritems()
                                   if key not in units
                                   and key not in quants)

    # Remove quant and unit from amr dict
    for q in quants:
        if q in amr_copy.keys():
            amr_copy.pop(q)
    for u in units:
        if u in amr_copy.keys():
            amr_copy.pop(u)

    # Remove quant and unit children from temporal-quantity in dict
    temp_quantities = [t[0] for t in quant_unit_tokens]
    for temp_quantity in temp_quantities:
        amr_copy[temp_quantity] = dict(
            (key, value) for key, value in amr_copy[temp_quantity].iteritems() if key != "quant" and key != "unit")

    # Add node_to_tokens for the temporal quantities with token as the "min" token spanned by the quantity and unit
    temporal_quantity_spans = [(t[0], min(t[3], t[4]), max(t[3], t[4]))
                               for t in quant_unit_tokens_align]
    temporal_quantity_spans = sorted(temporal_quantity_spans, key=itemgetter(1))

    tokens = sentence.split(" ")
    total_displacement = 0
    for temporal_quantity in temporal_quantity_spans:
        span_min = temporal_quantity[1]
        span_max = temporal_quantity[2]
        for n in amr_copy.node_to_tokens:
            amr_copy.node_to_tokens[n] = [t if int(t) < span_max
                                          else int(t) - (span_max - span_min)
                                          for t in amr_copy.node_to_tokens[n]]
        amr_copy.node_to_tokens[temporal_quantity[0]] = [span_min - total_displacement]
        tokens = [tokens[:(span_min - total_displacement)] +
                  [amr_copy.node_to_concepts[temporal_quantity[0]]] +
                  tokens[(span_max - total_displacement + 1):]][0]
        total_displacement = total_displacement + span_max - span_min
    sentence_copy = " ".join(t for t in tokens)
    return amr_copy, sentence_copy, quant_unit_tokens_align


def replace_quantities_default(amr, sentence, quantities):
    amr_copy = copy.deepcopy(amr)

    # Find all the "quantity" nodes.
    quantity_nodes = [k for k in amr_copy if k in amr_copy.node_to_concepts
                      and amr_copy.node_to_concepts[k] in quantities]

    if len(quantity_nodes) == 0:
        return amr, sentence, []

    # Find the "quant" and "unit" nodes corresponding to the temporal quantity.
    quant_unit_tokens = [(k, amr_copy[k]["quant"][0], amr_copy[k]["unit"][0])
                         if "quant" in amr_copy[k] and "unit" in amr_copy[k]
                         else (k, "", "")
                         for k in quantity_nodes]

    quant_unit_tokens_align = [(t[0], t[1], t[2],
                                int(amr_copy.node_to_tokens[t[1]][0][0]),
                                int(amr_copy.node_to_tokens[t[2]][0]))
                               if t[1] in amr_copy.node_to_tokens and
                                  t[2] in amr_copy.node_to_tokens
                               else (t, "", "", -1, -1)
                               for t in quant_unit_tokens]

    for t in quant_unit_tokens_align:
        if t[3] == -1 or t[4] == -1:
            raise ValueError("Error! Unaligned quantity or token for sentence %s" % sentence)

        else:
            if not (abs(t[3] - t[4]) == 1 or
                    (abs(t[3] - t[4]) == 2
                     and sentence.split(" ")[max(t[3], t[4]) - 1] == "@-@")):
                raise ValueError("Quant and unit not consecutive or separated by @-@ for sentence %s" % sentence)
    # Remove units from node_to_concepts
    units = [t[2] for t in quant_unit_tokens_align]
    amr_copy.node_to_concepts = dict((key, value) for key, value in amr_copy.node_to_concepts.iteritems()
                                     if key not in units)

    # Remove quant and unit from node_to_tokens
    quants = [t[1] for t in quant_unit_tokens_align]
    amr_copy.node_to_tokens = dict((key, value) for key, value in amr_copy.node_to_tokens.iteritems()
                                   if key not in units
                                   and key not in quants)

    # Remove quant and unit from amr dict
    for q in quants:
        if q in amr_copy.keys():
            amr_copy.pop(q)
    for u in units:
        if u in amr_copy.keys():
            if len(amr_copy[u]) != 0:
                raise ValueError("The unit node has additional children for sentence %s" % sentence)
            amr_copy.pop(u)

    # Remove quant and unit children from quantity in dict
    node_quantities = [t[0] for t in quant_unit_tokens]
    for quantity in node_quantities:
        amr_copy[quantity] = dict(
            (key, value) for key, value in amr_copy[quantity].iteritems() if key != "quant" and key != "unit")

    # Add node_to_tokens for the temporal quantities with token as the "min" token spanned by the quantity and unit
    quantity_spans = [(t[0], min(t[3], t[4]), max(t[3], t[4]))
                      for t in quant_unit_tokens_align]
    quantity_spans = sorted(quantity_spans, key=itemgetter(1))

    tokens = sentence.split(" ")
    total_displacement = 0
    for quantity in quantity_spans:
        span_min = quantity[1]
        span_max = quantity[2]
        for n in amr_copy.node_to_tokens:
            amr_copy.node_to_tokens[n] = [t if int(t) < span_max
                                          else int(t) - (span_max - span_min)
                                          for t in amr_copy.node_to_tokens[n]]
        amr_copy.node_to_tokens[quantity[0]] = [span_min - total_displacement]
        tokens = [tokens[:(span_min - total_displacement)] +
                  [amr_copy.node_to_concepts[quantity[0]]] +
                  tokens[(span_max - total_displacement + 1):]][0]
        total_displacement = total_displacement + span_max - span_min
    sentence_copy = " ".join(t for t in tokens)
    return amr_copy, sentence_copy, quant_unit_tokens_align


def replace_have_org_role(amr, relation_to_bubble_up):
    amr_copy = copy.deepcopy(amr)

    # get the unaligned have-org-role-91 nodes which have an ARG1 child, i.e. they can be replaced
    have_org_role_nodes = [k for k in amr_copy.node_to_concepts if amr_copy.node_to_concepts[k] == "have-org-role-91"
                           and k not in amr_copy.node_to_tokens
                           and amr_copy[k]
                           and relation_to_bubble_up in amr_copy[k]]
    if len(have_org_role_nodes) == 0:
        return amr, []

    amr_copy.node_to_concepts = dict(
        (k, v) for k, v in amr_copy.node_to_concepts.iteritems() if k not in have_org_role_nodes)
    for h in have_org_role_nodes:
        node = amr_copy.pop(h)
        new_node = node[relation_to_bubble_up]
        # add the have_org_role_children to the node corresponding to its ARG1 child
        node.pop(relation_to_bubble_up)
        if len(node) != 0:
            raise Exception("Amr %s has an have-org-role91 node with multiple children." % amr.to_amr_string())
        # update the parent of have_org_role
        for k in amr:
            for rel in amr_copy[k]:
                if amr_copy[k][rel][0] == h:
                    amr_copy[k].replace(rel, new_node)
    return amr_copy, have_org_role_nodes
