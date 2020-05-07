from typing import List, Tuple

from data_analysis.filtering.data_filtering import DataFiltering, CustomizedAMRDataFiltering
from data_analysis.filtering.filters import NoMissingAllignmentInfoFilter, TreeFilter, PerfectAlignmentFilter, \
    ProjectiveTreeFilter, \
    TokenToNodeAlignmentFilter
from data_extraction.dataset_reading_util import read_dataset_dict
from models.amr_graph import AMR
from models import amr_data
from preprocessing.preprocessing_steps import PreprocessingStep, HaveOrgPreprocessingStep, \
    NamedEntitiesPreprocessingStep, DateEntitiesPreprocessingStep, TemporalQuantitiesPreprocessingStep, \
    QuantitiesPreprocessingStep, apply_preprocessing_steps_on_instance


def generate_dataset_statistics(sentence_amr_str_triples: List[Tuple[str,str,str]], filters):
    # number of (sentence,amr) pairs that pass the amr parsing
    instances = 0
    # number of instances that pass the filters
    filtered_instances = 0

    sentence_amr_id = []
    amr_preprocessing_fails = 0

    for i in range(0, len(sentence_amr_str_triples)):
        (sentence, amr_str, amr_id) = sentence_amr_str_triples[i]
        # print("sentence: {0}\n amr_str: {1}\n".format(sentence, amr_str))

        try:

            amr = AMR.parse_string(amr_str)
            preprocessing_steps: List[PreprocessingStep] = [
                HaveOrgPreprocessingStep(),
                NamedEntitiesPreprocessingStep(),
                DateEntitiesPreprocessingStep(),
                TemporalQuantitiesPreprocessingStep(),
                QuantitiesPreprocessingStep()
            ]
            new_amr, new_sentence, _ = apply_preprocessing_steps_on_instance(amr, sentence,preprocessing_steps)

            custom_amr = amr_data.CustomizedAMR()
            custom_amr.create_custom_AMR(new_amr)
            sentence_amr_id.append((sentence, custom_amr, amr_id))

        except Exception as e:
            amr_preprocessing_fails += 1

    # apply filters
    filtering = CustomizedAMRDataFiltering(sentence_amr_id)
    for f in filters:
        filtering.add_filter(f)
    new_sentence_amr_pairs = filtering.execute()

    instances = len(sentence_amr_id)
    filtered_instances = len(new_sentence_amr_pairs)
    return instances, filtered_instances


def generate_statistics(split, dataset_name, data: List[Tuple[str, str, str]]):
    print(("Dataset: {0}".format(split+' '+dataset_name)))
    # NO FILTERS
    instances, filtered_instances = generate_dataset_statistics(data, [])
    print(("Filter: none, Total number of instances: {0}, Number of filtered instances: {1}".format(instances,
                                                                                                    filtered_instances)))

    # no missing alignment
    aligned_filters = [NoMissingAllignmentInfoFilter()]
    instances, filtered_instances = generate_dataset_statistics(data, aligned_filters)
    print(
        ("Filter: NoMissingAllignmentFilter, Total number of instances: {0}, Number of filtered instances: {1}".format(
            instances, filtered_instances)))

    # tree
    tree_filters = [TreeFilter()]
    instances, filtered_instances = generate_dataset_statistics(data, tree_filters)
    print(("Filter: TreeFilter, Total number of instances: {0}, Number of filtered instances: {1}".format(instances,
                                                                                                          filtered_instances)))

    # tree with no missing alignment
    aligned_treee_filters = [TreeFilter(), NoMissingAllignmentInfoFilter()]
    instances, filtered_instances = generate_dataset_statistics(data, aligned_treee_filters)
    print((
        "Filter: AlignedTreeFilter, Total number of instances: {0}, Number of filtered instances: {1}".format(instances,
                                                                                                              filtered_instances)))

    # from this point on, statistics are run on amrs that are aligned trees

    # aligned tree where token:node relation is 1:0..1
    aligned_treee_token1words_filters = [TreeFilter(), NoMissingAllignmentInfoFilter(), TokenToNodeAlignmentFilter(1)]
    instances, filtered_instances = generate_dataset_statistics(data, aligned_treee_token1words_filters)
    print((
        "Filter: Token1WordsFilter, Total number of instances: {0}, Number of filtered instances: {1}".format(instances,
                                                                                                              filtered_instances)))

    # aligned tree where token:node relation is 1:0..2
    aligned_treee_token2words_filters = [TreeFilter(), NoMissingAllignmentInfoFilter(), TokenToNodeAlignmentFilter(2)]
    instances, filtered_instances = generate_dataset_statistics(data, aligned_treee_token2words_filters)
    print((
        "Filter: Token2WordsFilter, Total number of instances: {0}, Number of filtered instances: {1}".format(instances,
                                                                                                              filtered_instances)))

    # aligned tree where token:node relation is 1:0..3
    aligned_treee_token3words_filters = [TreeFilter(), NoMissingAllignmentInfoFilter(), TokenToNodeAlignmentFilter(3)]
    instances, filtered_instances = generate_dataset_statistics(data, aligned_treee_token3words_filters)
    print((
        "Filter: Token3WordsFilter, Total number of instances: {0}, Number of filtered instances: {1}".format(instances,
                                                                                                              filtered_instances)))

    # perfectly aligned trees
    perfect_align_filters = [PerfectAlignmentFilter()]
    instances, filtered_instances = generate_dataset_statistics(data, perfect_align_filters)
    print(("Filter: PerfectlyAlignedAMRTree, Total number of instances: {0}, Number of filtered instances: {1}".format(
        instances, filtered_instances)))

    # aligned projective tree
    pojective_filters = [ProjectiveTreeFilter()]
    instances, filtered_instances = generate_dataset_statistics(data, pojective_filters)
    print(("Filter: ProjectiveAlignedTreee, Total number of instances: {0}, Number of filtered instances: {1}".format(
        instances, filtered_instances)))

    print("\n")


if __name__ == "__main__":

    splits = ["training", "dev", "test"]
    for split in splits:
        split_dataset_dict = read_dataset_dict(split)
        for dataset_name, data in split_dataset_dict.items():
            generate_statistics(split, dataset_name, data)
