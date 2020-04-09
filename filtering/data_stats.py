from data_filtering import DataFiltering
from data_extraction import input_file_parser
from models.amr_graph import AMR
from filters import NoMissingAllignmentInfoFilter, TreeFilter, TokenToNodeAlignmentFilter, ProjectiveTreeFilter, \
    PerfectAlignmentFilter
from preprocessing import TokensReplacer
from models import amr_data


def generate_dataset_statistics(file_path, filters):
    # number of (sentence,amr) pairs that pass the amr parsing
    instances = 0
    # number of instances that pass the filters
    filtered_instances = 0

    sentence_amr_id = []
    amr_preprocessing_fails = 0

    sentence_amr_str_triples = input_file_parser.extract_data_records(file_path)

    for i in range(0, len(sentence_amr_str_triples)):
        (sentence, amr_str, amr_id) = sentence_amr_str_triples[i]
        # print("sentence: {0}\n amr_str: {1}\n".format(sentence, amr_str))
        try:

            amr = AMR.parse_string(amr_str)

            (new_amr, _) = TokensReplacer.replace_have_org_role(amr, "ARG1")
            (new_amr, _) = TokensReplacer.replace_have_org_role(amr, "ARG2")

            (new_amr, new_sentence, named_entities) = TokensReplacer.replace_named_entities(amr, sentence)
            (new_amr, new_sentence, date_entities) = TokensReplacer.replace_date_entities(new_amr, new_sentence)
            (new_amr, new_sentence, _) = TokensReplacer.replace_temporal_quantities(new_amr, new_sentence)
            (new_amr, new_sentence, _) = TokensReplacer.replace_quantities_default(new_amr, new_sentence,
                                                                                   ['monetary-quantity',
                                                                                    'mass-quantity',
                                                                                    'energy-quantity',
                                                                                    'distance-quantity',
                                                                                    'volume-quantity',
                                                                                    'power-quantity'
                                                                                    ])

            custom_amr = amr_data.CustomizedAMR()
            custom_amr.create_custom_AMR(new_amr)
            sentence_amr_id.append((sentence, new_amr, custom_amr, amr_id))

        except Exception as e:
            amr_preprocessing_fails += 1

    # apply filters
    filtering = DataFiltering(sentence_amr_id)
    for f in filters:
        filtering.add_filter(f)
    new_sentence_amr_pairs = filtering.execute()

    instances = len(sentence_amr_id)
    filtered_instances = len(new_sentence_amr_pairs)
    return instances, filtered_instances


def generate_statistics(file_path):
    print(("Dataset: {0}".format(file_path)))

    # NO FILTERS
    instances, filtered_instances = generate_dataset_statistics(file_path, [])
    print(("Filter: none, Total number of instances: {0}, Number of filtered instances: {1}".format(instances,
                                                                                                   filtered_instances)))

    # no missing alignment
    aligned_filters = [NoMissingAllignmentInfoFilter()]
    instances, filtered_instances = generate_dataset_statistics(file_path, aligned_filters)
    print(("Filter: NoMissingAllignmentFilter, Total number of instances: {0}, Number of filtered instances: {1}".format(
        instances, filtered_instances)))

    # tree
    tree_filters = [TreeFilter()]
    instances, filtered_instances = generate_dataset_statistics(file_path, tree_filters)
    print(("Filter: TreeFilter, Total number of instances: {0}, Number of filtered instances: {1}".format(instances,
                                                                                                         filtered_instances)))

    # tree with no missing alignment
    aligned_treee_filters = [TreeFilter(), NoMissingAllignmentInfoFilter()]
    instances, filtered_instances = generate_dataset_statistics(file_path, aligned_treee_filters)
    print((
        "Filter: AlignedTreeFilter, Total number of instances: {0}, Number of filtered instances: {1}".format(instances,
                                                                                                              filtered_instances)))

    # from this point on, statistics are run on amrs that are aligned trees

    # aligned tree where token:node relation is 1:0..1
    aligned_treee_token1words_filters = [TreeFilter(), NoMissingAllignmentInfoFilter(), TokenToNodeAlignmentFilter(1)]
    instances, filtered_instances = generate_dataset_statistics(file_path, aligned_treee_token1words_filters)
    print((
        "Filter: Token1WordsFilter, Total number of instances: {0}, Number of filtered instances: {1}".format(instances,
                                                                                                              filtered_instances)))

    # aligned tree where token:node relation is 1:0..2
    aligned_treee_token2words_filters = [TreeFilter(), NoMissingAllignmentInfoFilter(), TokenToNodeAlignmentFilter(2)]
    instances, filtered_instances = generate_dataset_statistics(file_path, aligned_treee_token2words_filters)
    print((
        "Filter: Token2WordsFilter, Total number of instances: {0}, Number of filtered instances: {1}".format(instances,
                                                                                                              filtered_instances)))

    # aligned tree where token:node relation is 1:0..3
    aligned_treee_token3words_filters = [TreeFilter(), NoMissingAllignmentInfoFilter(), TokenToNodeAlignmentFilter(3)]
    instances, filtered_instances = generate_dataset_statistics(file_path, aligned_treee_token3words_filters)
    print((
        "Filter: Token3WordsFilter, Total number of instances: {0}, Number of filtered instances: {1}".format(instances,
                                                                                                              filtered_instances)))

    # perfectly aligned trees
    perfect_align_filters = [PerfectAlignmentFilter()]
    instances, filtered_instances = generate_dataset_statistics(file_path, perfect_align_filters)
    print(("Filter: PerfectlyAlignedAMRTree, Total number of instances: {0}, Number of filtered instances: {1}".format(
        instances, filtered_instances)))

    # aligned projective tree
    pojective_filters = [ProjectiveTreeFilter()]
    instances, filtered_instances = generate_dataset_statistics(file_path, pojective_filters)
    print(("Filter: ProjectiveAlignedTreee, Total number of instances: {0}, Number of filtered instances: {1}".format(
        instances, filtered_instances)))

    print("\n")


splits = ["training", "dev", "test"]
data_sets = {"training": ["bolt", "cctv", "dfa", "dfb", "guidelines", "mt09sdl", "proxy", "wb", "xinhua"],
             "dev": ["bolt", "consensus", "dfa", "proxy", "xinhua"],
             "test": ["bolt", "consensus", "dfa", "proxy", "xinhua"]}

for split in splits:
    for data_set in data_sets[split]:
        my_file_path = '../resources/alignments/split/' + split + "/" + "deft-p2-amr-r1-alignments-" + split + "-" + data_set + ".txt"
        generate_statistics(my_file_path)
