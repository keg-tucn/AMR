from itertools import groupby

from data_extraction.dependency_extractor import DependencyExtractor
from stanford_parser.parser import Parser


class StanfordDependencyExtractor(DependencyExtractor):
    stanford_parser = Parser()

    @classmethod
    def extract_dependencies(cls, sentence):
        indices = StanfordDependencyExtractor._split_with_indices(sentence)
        dependencies = cls.stanford_parser.parseToStanfordDependencies(sentence.strip())
        dependencies_dict = {}
        for (rel, gov, dep) in dependencies.dependencies:
            if dep.start in list(indices.keys()) and gov.start in list(indices.keys()):
                dependencies_dict[indices[dep.start]] = (indices[gov.start], rel)

        return dependencies_dict

    @classmethod
    def _split_with_indices(cls, string, split_char=" "):
        p = 0
        start_indices = {}
        current_token = 0
        for k, g in groupby(string, lambda x: x == split_char):
            q = p + sum(1 for i in g)
            if not k:
                start_indices[p] = current_token
                current_token += 1
            p = q
        return start_indices
