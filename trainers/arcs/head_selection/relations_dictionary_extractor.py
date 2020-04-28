from typing import List

from data_extraction import input_file_parser
from models.amr_data import CustomizedAMR
from models.amr_graph import AMR

UNKNOWN_RELATION = 'unk-rel'


def __get_concept_for_var(amr: CustomizedAMR, variable):
    if variable in amr.amr_graph.node_to_concepts.keys():
        concept = amr.amr_graph.node_to_concepts[variable]
    else:
        # string literal (is it the only case? should I treat it differently??)
        concept = variable
    return concept


def extract_relation_counts_dict(amrs: List[CustomizedAMR]):
    """Go through the dataset consisting of the AMRs given through amrs
    and create a dictionary of the form (c1, c2) -> {rel1: count, rel2: count, rel3:count...}
    For example:
    (advocate-01,it) -> {ARG1: 75, ARG2: 20, ARG3:5}
    """
    relation_counts = {}
    for amr in amrs:
        for child_var, parent_var in amr.relations_dict.keys():
            relation_info = amr.relations_dict[(child_var, parent_var)]
            relation = relation_info[0]
            if relation != '':
                amr_graph: AMR
                # todo: find how this will be affected by preprocessing
                amr_graph = amr.amr_graph
                child = __get_concept_for_var(amr, child_var)
                parent = __get_concept_for_var(amr, parent_var)
                if (parent, child) not in relation_counts.keys():
                    relation_counts[(parent, child)] = {}
                if relation not in relation_counts[(parent, child)].keys():
                    relation_counts[(parent, child)][relation] = 1
                else:
                    relation_counts[(parent, child)][relation] += 1
    return relation_counts


# maybe should be in some util file
def extract_custom_amrs_from_file(file_path: str):
    sentence_amr_triples = input_file_parser.extract_data_records(file_path)
    custom_amrs = []
    for _, amr_str, _ in sentence_amr_triples:
        amr = AMR.parse_string(amr_str)
        custom_amr: CustomizedAMR = CustomizedAMR()
        custom_amr.create_custom_AMR(amr)
        custom_amrs.append(custom_amr)
    return custom_amrs


def extract_relation_dict(file_paths: List[str]):
    """
    Go through the dataset consisting of the files given through file_paths
    and create a dictionary of the form (c1, c2) -> relation
    where relation is the relation with the highest occurence frequency between c1 & c2
    For example, if between
    (advocate-01,it) -> 75% of the time relation "ARG1"
    (advocate-01,it) -> 20% of the time relation "ARG2"
    (advocate-01,it) -> 5% of the time relation "ARG3"
    in the dictionary we will have
    (advocate-01,it) -> ARG1
    """
    # TODO: figure out what to do in case of preprocessing the AMRs
    # 1) read the amrs from the files
    amrs: List[CustomizedAMR] = []
    for file_path in file_paths:
        amrs_from_file = extract_custom_amrs_from_file(file_path)
        amrs = amrs + amrs_from_file

    # 2) extract relation_dict
    relation_dict = {}
    relation_counts_dict = extract_relation_counts_dict(amrs)
    for concept_pair, relation_counts in relation_counts_dict.items():
        max_occurence = 0
        max_relation = UNKNOWN_RELATION
        for relation in relation_counts.keys():
            if relation_counts[relation] > max_occurence:
                max_occurence = relation_counts[relation]
                max_relation = relation
        relation_dict[concept_pair] = max_relation
    return relation_dict


def get_relation_between_concepts(relation_dict, parent: str, child: str):
    if (parent, child) in relation_dict.keys():
        return relation_dict[(parent, child)]
    return UNKNOWN_RELATION
