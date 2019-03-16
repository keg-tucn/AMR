from itertools import groupby

from stanford_parser.parser import Parser


class DependencyExtractor:
    stanford_parser = Parser()

    @classmethod
    def extract_dependencies(cls, sentence):
        indices = DependencyExtractor._split_with_indices(sentence)
        dependencies = DependencyExtractor.stanford_parser.parseToStanfordDependencies(sentence.strip())
        dependencies_dict = {}
        for (rel, gov, dep) in dependencies.dependencies:
            if dep.start in indices.keys() and gov.start in indices.keys():
                dependencies_dict[indices[dep.start]] = (indices[gov.start], rel)
        return dependencies_dict

    @classmethod
    def _split_with_indices(cls, string, split_char=' '):
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
