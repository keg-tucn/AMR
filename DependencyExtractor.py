from os import path
import sys
import logging
from itertools import groupby
from stanford_parser.parser import Parser
sys.path.append(path.abspath('./stanford_parser'))

def splitWithIndices(string, splitChar=' '):
    p = 0
    start_indices = {}
    current_token = 0
    for k, g in groupby(string, lambda x: x == splitChar):
        q = p + sum(1 for i in g)
        if not k:
            start_indices[p] = current_token
            current_token += 1
        p = q
    return start_indices


def extract_dependencies(sentence):
    parser = Parser()
    indices = splitWithIndices(sentence)
    dependencies = parser.parseToStanfordDependencies(sentence.strip())
    dependencies_dict = {}
    for (rel, gov, dep) in dependencies.dependencies:
        if dep.start in indices.keys() and gov.start in indices.keys():
            dependencies_dict[indices[dep.start]] = (indices[gov.start], rel)
    return dependencies_dict

