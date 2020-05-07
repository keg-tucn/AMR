from typing import List, Tuple

from amr_util.dataset_dictionary_util import map_to_amr_dataset_dict, apply_preprocessing_on_amr_dataset_dict, \
    map_from_amr_to_custom_amr_dataset_dict
from data_analysis.concepts_analysis import filter_dataset_dict
from data_analysis.dataset_analysis import DatasetAnalysis
from data_analysis.filtering.filters import CanExtractOrderedConceptsFilter
from data_extraction.dataset_reading_util import read_dataset_dict
from models.amr_data import CustomizedAMR
from models.amr_graph import AMR
from preprocessing.preprocessing_steps import NamedEntitiesPreprocessingStep, DateEntitiesPreprocessingStep, \
    TemporalQuantitiesPreprocessingStep, QuantitiesPreprocessingStep


class ArcsAnalysis(DatasetAnalysis):

    def __get_concept_name(self, amr: AMR, node: str):
        if node in amr.node_to_concepts.keys():
            concept = amr.node_to_concepts[node]
        else:
            # nodes without var
            concept = node
        return concept

    def get_arcs(self, data: List[Tuple[str, CustomizedAMR, str]]):
        arcs = set()
        for (sentence, custom_amr, amr_id) in data:
            amr = custom_amr.amr_graph
            for parent_node, children_arcs in amr.items():
                parent_concept = self.__get_concept_name(amr, parent_node)
                for children_arc in children_arcs.values():
                    child_node = children_arc[0]
                    child_concept = self.__get_concept_name(amr, child_node)
                    arcs.add((parent_concept, child_concept))
        return arcs

    def create_property_set(self, dataset_dictionary):

        arcs_set_per_dataset = {}
        for dataset, data in dataset_dictionary.items():
            arcs = self.get_arcs(data)
            arcs_set_per_dataset[dataset] = arcs
        return arcs_set_per_dataset

    # def create_tables(self):
    #
    #     # read data
    #     training_dataset_dict = read_custom_amr_dataset_dict('training')
    #     dev_dataset_dict = read_custom_amr_dataset_dict('dev')
    #
    #     # create concepts <-> training table
    #     self.create_comparative_analysis('tables/all_arcs_training_vs_dev.xlsx',
    #                                      'arcs',
    #                                      training_dataset_dict, dev_dataset_dict,
    #                                      'training', 'dev')

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
            tablename = 'tables/arcs/all_arcs_' + prep_name + '_training_vs_dev.xlsx'
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
                                             'arcs',
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
            tablename = 'tables/arcs/can_be_ordered_concepts_' + prep_name + '_training_vs_dev.xlsx'
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
                                             'arcs',
                                             filterd_training_dataset_dict, filterd_dev_dataset_dict,
                                             'training', 'dev')


if __name__ == "__main__":
    arcsAnalysis = ArcsAnalysis()
    # arcsAnalysis.create_tables()
    arcsAnalysis.create_tables_with_ordered_concepts_filter()
