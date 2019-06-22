from itertools import groupby
from stanford_parser.parser import Parser
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tokenize.stanford import StanfordTokenizer
import spacy

from definitions import *


class DependencyExtractor:
    def __init__(self):
        pass

    stanford_parser = Parser()
    stanford_parser_2 = StanfordDependencyParser(path_to_jar=STANFORD_PARSER_JAR,
                                                 path_to_models_jar=STANFORD_PARSER_MODEL)
    tokenizer = StanfordTokenizer(path_to_jar=STANFORD_POSTAGGER_JAR)
    spacy_nlp_processor = spacy.load("en_core_web_sm")

    @classmethod
    def extract_dependencies(cls, sentence):
        indices = DependencyExtractor._split_with_indices(sentence)
        dependencies = cls.stanford_parser.parseToStanfordDependencies(sentence.strip())
        dependencies_dict = {}
        for (rel, gov, dep) in dependencies.dependencies:
            if dep.start in indices.keys() and gov.start in indices.keys():
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

    @classmethod
    def extract_dependencies_2(cls, sentence):
        result = cls.stanford_parser_2.raw_parse(sentence)
        dependencies = result.next()

        tokens = cls.tokenizer.tokenize(sentence)

        deps_dict = {}
        for dep in list(dependencies.triples()):
            gov_token = tokens.index(dep[0][0])
            dep_token = tokens.index(dep[2][0])
            dep_type = dep[1].encode("utf-8")
            # if dep_token not in deps_dict:
            #    deps_dict[dep_token] = []
            # deps_dict[dep_token].append((gov_token, dep_type))
            deps_dict[dep_token] = (gov_token, dep_type)
        return deps_dict

    @classmethod
    def extract_dependencies_spacy(cls, sentence):
        unicode_sentence = sentence.decode("utf-8")
        doc = cls.spacy_nlp_processor(unicode_sentence)

        deps_dict = {}
        for token in doc:
            gov_token = token.i
            dep_token = token.head.i
            dep_type = token.dep_.encode("utf-8")
            # if gov_token not in deps_dict:
            #     deps_dict[gov_token] = []
            # deps_dict[gov_token].append((dep_token, dep_type))
            deps_dict[gov_token] = (dep_token, dep_type)
        return deps_dict


if __name__ == "__main__":
    sentence = "Autonomous cars shift insurance liability toward manufacturers"
    unicode_sentence = sentence.decode("utf-8")

    tokens = DependencyExtractor.tokenizer.tokenize(sentence)

    print "Sentence: %s" % sentence
    tokens_str = ""
    for t, i in zip(tokens, range(len(tokens))):
        tokens_str += "%d:%s " % (i, t)
    print tokens_str

    print "\nOriginal version Stanford Parser"
    print DependencyExtractor.extract_dependencies(sentence)

    print "\nUpdated version Stanford Parser"
    print DependencyExtractor.extract_dependencies_2(sentence)

    print "\nSpaCy Parser"
    print DependencyExtractor.extract_dependencies_spacy(unicode_sentence)

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(u"Autonomous cars shift insurance liability toward manufacturers")

    print ""
    for token in doc:
        print "%d:%s %s %d:%s" % (token.i, token.text, token.dep_, token.head.i, token.head.text)
