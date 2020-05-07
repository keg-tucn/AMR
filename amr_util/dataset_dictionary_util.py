from typing import List

from models.amr_data import CustomizedAMR
from models.amr_graph import AMR
from preprocessing.preprocessing_steps import apply_preprocessing_steps_on_amr_list, PreprocessingStep


def map_to_amr_dataset_dict(dataset_dict):
    """
    Takes as input a dictionary of the form:
        {dataset: [sentence, amr_str, amr_id]}
    Outputs a dictionary of the form:
        {dataset: [sentence, amr: AMR, amr_id]}
    """
    amr_dataset_dict = {}
    for dataset, data in dataset_dict.items():
        new_format_data = []
        for data_item in data:
            amr_str = data_item[1]
            # TODO: util for amr_str -> custom_amr
            amr = AMR.parse_string(amr_str)
            new_format_data.append((data_item[0], amr, data_item[2]))
        amr_dataset_dict[dataset] = new_format_data
    return amr_dataset_dict


def map_to_custom_amr_dataset_dict(dataset_dict):
    """
    Takes as input a dictionary of the form:
        {dataset: [sentence, amr_str, amr_id]}
    Outputs a dictionary of the form:
        {dataset: [sentence, custom_amr, amr_id]}
    """
    custom_amr_dataset_dict = {}
    for dataset, data in dataset_dict.items():
        new_format_data = []
        for data_item in data:
            amr_str = data_item[1]
            # TODO: util for amr_str -> custom_amr
            amr = AMR.parse_string(amr_str)
            custom_amr = CustomizedAMR()
            custom_amr.create_custom_AMR(amr)
            new_format_data.append((data_item[0], custom_amr, data_item[2]))
        custom_amr_dataset_dict[dataset] = new_format_data
    return custom_amr_dataset_dict


def map_from_amr_to_custom_amr_dataset_dict(dataset_dict):
    """
    Takes as input a dictionary of the form:
        {dataset: [sentence, amr, amr_id]}
    Outputs a dictionary of the form:
        {dataset: [sentence, custom_amr, amr_id]}
    """
    custom_amr_dataset_dict = {}
    for dataset, data in dataset_dict.items():
        new_format_data = []
        for data_item in data:
            amr = data_item[1]
            custom_amr = CustomizedAMR()
            custom_amr.create_custom_AMR(amr)
            new_format_data.append((data_item[0], custom_amr, data_item[2]))
        custom_amr_dataset_dict[dataset] = new_format_data
    return custom_amr_dataset_dict


def apply_preprocessing_on_amr_dataset_dict(dataset_dict, preprocessing_steps: List[PreprocessingStep]):

    preprocessed_amrs_dict = {}
    preprocessing_metadata_dict = {}
    for dataset, data in dataset_dict.items():
        preprocessed_amrs, preprocessing_metadatas = \
            apply_preprocessing_steps_on_amr_list(data, preprocessing_steps)
        preprocessed_amrs_dict[dataset] = preprocessed_amrs
        preprocessing_metadata_dict[dataset] = preprocessing_metadatas
    return preprocessed_amrs_dict, preprocessing_metadata_dict
