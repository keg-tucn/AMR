from abc import abstractmethod


class DependencyExtractor:
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def extract_dependencies(cls, sentence):
        pass
