from typing import List, Tuple

from models.amr_graph import AMR
from pre_post_processing.standford_pre_post_processing import train_pre_processing
from preprocessing.TokensReplacer import replace_date_entities, replace_named_entities, replace_temporal_quantities, \
    replace_have_org_role, replace_quantities_default
from preprocessing.preprocessing_metadata import PreprocessingMetadata

DATE_ENTITIES_STEP = 'data-entities-step'
NAMED_ENTITIES_STEP = 'named-entities-step'
TEMPORAL_QUANTITIES_STEP = 'temporal-quantities-step'
QUANTITIES_STEP = 'quantities-step'
HAVE_ORG_STEP = 'have-org-step'
STANFORD_NER_STEP = 'stanford-ner-step'


class PreprocessingStep:

    def get_name(self):
        raise NotImplemented()

    def apply_step(self, amr, sentence):
        raise NotImplemented()

    def preprocess(self, amr: AMR, sentence: str, metadata: PreprocessingMetadata):
        new_amr, new_sentence, step_metadata = self.apply_step(amr, sentence)
        metadata.add_step(self.get_name(), step_metadata)
        return new_amr, new_sentence, metadata


class DateEntitiesPreprocessingStep(PreprocessingStep):

    def get_name(self):
        return DATE_ENTITIES_STEP

    def apply_step(self, amr, sentence):
        return replace_date_entities(amr, sentence)


class NamedEntitiesPreprocessingStep(PreprocessingStep):

    def get_name(self):
        return NAMED_ENTITIES_STEP

    def apply_step(self, amr, sentence):
        return replace_named_entities(amr, sentence)


class TemporalQuantitiesPreprocessingStep(PreprocessingStep):

    def get_name(self):
        return TEMPORAL_QUANTITIES_STEP

    def apply_step(self, amr, sentence):
        return replace_temporal_quantities(amr, sentence)


class QuantitiesPreprocessingStep(PreprocessingStep):

    def get_name(self):
        return QUANTITIES_STEP

    def apply_step(self, amr, sentence):
        return replace_quantities_default(amr,
                                          sentence,
                                          ['monetary-quantity',
                                           'mass-quantity',
                                           'energy-quantity',
                                           'distance-quantity',
                                           'volume-quantity',
                                           'power-quantity'
                                           ]
                                          )


class HaveOrgPreprocessingStep(PreprocessingStep):

    def get_name(self):
        return HAVE_ORG_STEP

    def apply_step(self, amr, sentence):
        new_amr, have_org_role_nodes_arg1 = replace_have_org_role(amr, 'ARG1')
        new_amr, have_org_role_nodes_arg2 = replace_have_org_role(new_amr, 'ARG2')
        have_org_metadata = {'ARG1': have_org_role_nodes_arg1, 'ARG2': have_org_role_nodes_arg2}
        return new_amr, sentence, have_org_metadata


class StandfordNerTaggerPreprocessingStep(PreprocessingStep):

    def get_name(self):
        return STANFORD_NER_STEP

    def apply_step(self, amr, sentence):
        new_amr, new_sentence, metadata = (amr, sentence)
        return new_amr, sentence, metadata


def apply_preprocessing_steps_on_instance(amr: AMR, sentence: str,
                                          preprocessing_steps: List[PreprocessingStep],
                                          throwExceptions=False):
    preprocessing_metadata: PreprocessingMetadata = PreprocessingMetadata()
    for preprocessing_step in preprocessing_steps:
        amr, sentence, preprocessing_metadata = preprocessing_step.preprocess(amr=amr,
                                                                              sentence=sentence,
                                                                              metadata=preprocessing_metadata)
    return amr, sentence, preprocessing_metadata


def apply_preprocessing_steps_on_amr_list(amrs: Tuple[List[str], List[AMR], List[str]],
                                          preprocessing_steps: List[PreprocessingStep],
                                          throwExceptions=False):
    # intialize exception counts
    exception_counts = {}
    for preprocessing_step in preprocessing_steps:
        exception_counts[preprocessing_step.get_name()] = 0
    preprocessed_amrs = []
    preprocessing_metadatas = []
    for sentence, amr, amr_id in amrs:
        try:
            new_amr, new_sentence, preprocessing_metadata = apply_preprocessing_steps_on_instance(amr, sentence,
                                                                                                  preprocessing_steps)
            preprocessed_amrs.append((new_sentence, new_amr, amr_id))
            preprocessing_metadatas.append(preprocessing_metadata)
        except Exception as e:
            exception_counts[preprocessing_step.get_name()] += 1
            if throwExceptions:
                raise e
    return preprocessed_amrs, preprocessing_metadatas
