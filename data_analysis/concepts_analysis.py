from typing import List, Tuple

from amr_util.dataset_dictionary_util import map_to_custom_amr_dataset_dict, map_to_amr_dataset_dict, \
    apply_preprocessing_on_amr_dataset_dict, map_from_amr_to_custom_amr_dataset_dict
from data_analysis.dataset_analysis import DatasetAnalysis
from data_analysis.filtering.data_filtering import CustomizedAMRDataFiltering
from data_analysis.filtering.filters import CanExtractOrderedConceptsFilter
from data_extraction.dataset_reading_util import read_dataset_dict
from models.amr_data import CustomizedAMR
from preprocessing.preprocessing_steps import PreprocessingStep, HaveOrgPreprocessingStep, \
    NamedEntitiesPreprocessingStep, DateEntitiesPreprocessingStep, TemporalQuantitiesPreprocessingStep, \
    QuantitiesPreprocessingStep


def filter_dataset_dict(dataset_dict, filters):
    """
    dataset_dict: {dataset: [sentence, custom_amr, amr_id]
    """
    filtered_dataset_dict = {}
    for dataset, data in dataset_dict.items():
        filtering = CustomizedAMRDataFiltering(data)
        for f in filters:
            filtering.add_filter(f)
        filtered_data = filtering.execute()
        filtered_dataset_dict[dataset] = filtered_data
    return filtered_dataset_dict


class ConceptsAnalysis(DatasetAnalysis):

    def get_concepts(self, data: List[Tuple[str, CustomizedAMR, str]]):
        concepts = set()
        for (sentence, custom_amr, amr_id) in data:
            amr = custom_amr.amr_graph
            for node, _ in amr.items():
                if node in amr.node_to_concepts.keys():
                    concept = amr.node_to_concepts[node]
                else:
                    # nodes without var
                    concept = node
                concepts.add(concept)
        return concepts

    def create_property_set(self, dataset_dictionary):

        concepts_set_per_dataset = {}
        for dataset, data in dataset_dictionary.items():
            concepts = self.get_concepts(data)
            concepts_set_per_dataset[dataset] = concepts
        return concepts_set_per_dataset

    def create_tables(self):

        # read data
        training_dataset_dict = read_dataset_dict('training')
        dev_dataset_dict = read_dataset_dict('dev')

        # apply processing steps
        preprocessing_steps_dict = {
            'no_prep': [],
            'ne_prep': [NamedEntitiesPreprocessingStep()],
            'de_prep': [DateEntitiesPreprocessingStep()],
            'tq_prep': [TemporalQuantitiesPreprocessingStep()],
            'q_prep': [QuantitiesPreprocessingStep()]
            # 'all_prep': [NamedEntitiesPreprocessingStep(),
            #              DateEntitiesPreprocessingStep(),
            #              TemporalQuantitiesPreprocessingStep(),
            #              QuantitiesPreprocessingStep()]
        }

        for prep_name, preprocessing_steps in preprocessing_steps_dict.items():
            tablename = 'tables/concepts/all_concepts_' + prep_name + '_training_vs_dev.xlsx'
            print('Start creating table ' + tablename)

            # preprocess
            amr_training_dataset = map_to_amr_dataset_dict(training_dataset_dict)
            amr_dev_dataset = map_to_amr_dataset_dict(dev_dataset_dict)
            prep_amr_training_dataset, _ = apply_preprocessing_on_amr_dataset_dict(amr_training_dataset,
                                                                                   preprocessing_steps)
            prep_amr_dev_dataset, _ = apply_preprocessing_on_amr_dataset_dict(amr_dev_dataset, preprocessing_steps)

            custom_amr_training_dataset = map_from_amr_to_custom_amr_dataset_dict(prep_amr_training_dataset)
            custom_amr_dev_dataset_dict = map_from_amr_to_custom_amr_dataset_dict(prep_amr_dev_dataset)

            # create concepts <-> training table
            self.create_comparative_analysis(tablename,
                                             'concepts',
                                             custom_amr_training_dataset, custom_amr_dev_dataset_dict,
                                             'training', 'dev')

    def create_tables_with_ordered_concepts_filter(self):

        # read data
        training_dataset_dict = read_dataset_dict('training')
        dev_dataset_dict = read_dataset_dict('dev')

        # apply processing steps
        preprocessing_steps_dict = {
            'no_prep': [],
            'ne_prep': [NamedEntitiesPreprocessingStep()],
            'de_prep': [DateEntitiesPreprocessingStep()],
            'tq_prep': [TemporalQuantitiesPreprocessingStep()],
            'q_prep': [QuantitiesPreprocessingStep()]
            # 'all_prep': [NamedEntitiesPreprocessingStep(),
            #              DateEntitiesPreprocessingStep(),
            #              TemporalQuantitiesPreprocessingStep(),
            #              QuantitiesPreprocessingStep()]
        }

        for prep_name, preprocessing_steps in preprocessing_steps_dict.items():
            tablename = 'tables/concepts/can_be_ordered_concepts_' + prep_name + '_training_vs_dev.xlsx'
            print('Start creating table ' + tablename)

            # preprocess
            amr_training_dataset = map_to_amr_dataset_dict(training_dataset_dict)
            amr_dev_dataset = map_to_amr_dataset_dict(dev_dataset_dict)
            prep_amr_training_dataset, _ = apply_preprocessing_on_amr_dataset_dict(amr_training_dataset,
                                                                                   preprocessing_steps)
            prep_amr_dev_dataset, _ = apply_preprocessing_on_amr_dataset_dict(amr_dev_dataset, preprocessing_steps)

            custom_amr_training_dataset = map_from_amr_to_custom_amr_dataset_dict(prep_amr_training_dataset)
            custom_amr_dev_dataset_dict = map_from_amr_to_custom_amr_dataset_dict(prep_amr_dev_dataset)

            # filter data (only concepts that can be extracted)
            filterd_training_dataset_dict = filter_dataset_dict(custom_amr_training_dataset,
                                                                [CanExtractOrderedConceptsFilter()])
            filterd_dev_dataset_dict = filter_dataset_dict(custom_amr_dev_dataset_dict,
                                                           [CanExtractOrderedConceptsFilter()])

            # create concepts <-> training table
            self.create_comparative_analysis(tablename,
                                             'concepts',
                                             filterd_training_dataset_dict, filterd_dev_dataset_dict,
                                             'training', 'dev')


if __name__ == "__main__":
    conceptsAnalysis = ConceptsAnalysis()
    conceptsAnalysis.create_tables()
    conceptsAnalysis.create_tables_with_ordered_concepts_filter()
