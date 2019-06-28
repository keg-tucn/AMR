from data_extraction.spacy_dependency_extractor import SpaCyDependencyExtractor
from data_extraction.standord_dependency_extractor import StanfordDependencyExtractor
from models.parameters import ParserParameters


def get_dependency_extractor_implementation(parser_parameters=ParserParameters()):
    dependency_extractor_implementation = None

    deps_source = parser_parameters.deps_source

    if deps_source == "stanford":
        dependency_extractor_implementation = StanfordDependencyExtractor
    elif deps_source == "spaCy":
        dependency_extractor_implementation = SpaCyDependencyExtractor

    return dependency_extractor_implementation


if __name__ == "__main__":
    sentence = "Autonomous cars shift insurance liability toward manufacturers"
    unicode_sentence = sentence.decode("utf-8")

    print "\nOriginal version Stanford Parser"
    print StanfordDependencyExtractor.extract_dependencies(sentence)

    print "\nSpaCy Parser"
    print SpaCyDependencyExtractor.extract_dependencies(unicode_sentence)
