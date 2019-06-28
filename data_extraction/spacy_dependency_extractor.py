import spacy

from data_extraction.dependency_extractor import DependencyExtractor


class SpaCyDependencyExtractor(DependencyExtractor):
    spacy_nlp_processor = spacy.load("en_core_web_sm")

    @classmethod
    def extract_dependencies(cls, sentence):
        unicode_sentence = sentence.decode("utf-8")
        doc = cls.spacy_nlp_processor(unicode_sentence)

        deps_dict = {}
        for token in doc:
            gov_token = token.i
            dep_token = token.head.i
            dep_type = token.dep_.encode("utf-8")
            deps_dict[gov_token] = (dep_token, dep_type)
        return deps_dict
