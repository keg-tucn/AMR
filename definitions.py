import configparser
import os

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE_NAME = "properties.ini"

CONFIG_FILE_PATH = PROJECT_ROOT_DIR + "/" + CONFIG_FILE_NAME

config = configparser.ConfigParser()
config.read(CONFIG_FILE_PATH)

STANFORD_DEP_PARSER_HOME = PROJECT_ROOT_DIR + "/" + config.get("STANFORD_DEP_PARSER", "PARSER_HOME")

STANFORD_PARSER_JAR = PROJECT_ROOT_DIR + "/" + config.get("STANFORD_PARSER", "PARSER_JAR")
STANFORD_PARSER_MODEL = PROJECT_ROOT_DIR + "/" + config.get("STANFORD_PARSER", "PARSER_MODEL")

STANFORD_NER_JAR = PROJECT_ROOT_DIR + "/" + config.get("STANFORD_NER", "PARSER_JAR")
STANFORD_NER_MODEL = PROJECT_ROOT_DIR + "/" + config.get("STANFORD_NER", "PARSER_MODEL")

STANFORD_POSTAGGER_JAR = PROJECT_ROOT_DIR + "/" + config.get("STANFORD_POS_TAGGER", "PARSER_JAR")

TOKENIZER_PATH = PROJECT_ROOT_DIR + "/" + config.get("TOKENIZER", "TOKENIZER_PATH")

GLOVE_EMBEDDINGS = PROJECT_ROOT_DIR + "/" + config.get("GLOVE_EMBEDDINGS", "EMBEDDINGS_PATH")

PROPBANK_FRAMES = PROJECT_ROOT_DIR + "/" + config.get("PROPBANK", "FRAMES_PATH")
PROPBANK_DUMP = PROJECT_ROOT_DIR + "/" + config.get("PROPBANK", "DUMP_PATH")

NOMBANK_FRAMES = PROJECT_ROOT_DIR + "/" + config.get("NOMBANK", "FRAMES_PATH")
NOMBANK_DUMP = PROJECT_ROOT_DIR + "/" + config.get("NOMBANK", "DUMP_PATH")

AMR_ALIGNMENTS_SPLIT = PROJECT_ROOT_DIR + "/" + config.get("AMR_ALIGNMENTS", "SPLIT")
AMR_ALIGNMENTS_UNSPLIT = PROJECT_ROOT_DIR + "/" + config.get("AMR_ALIGNMENTS", "UNSPLIT")

JAMR_ALIGNMENTS_SPLIT = PROJECT_ROOT_DIR + "/" + config.get("JAMR_ALIGNMENTS", "SPLIT")
MERGED_ALIGNMENTS_SPLIT = PROJECT_ROOT_DIR + "/" + config.get("MERGED_ALIGNMENTS", "SPLIT")

CONCEPTS_RELATIONS_DICT = PROJECT_ROOT_DIR + "/" + config.get("CONCEPTS_RELATIONS", "DICT_PATH")

TRAINED_MODELS_DIR = PROJECT_ROOT_DIR + "/" + config.get("TRAINED_MODELS", "MODELS_PATH")

RESULT_PLOTS_DIR = PROJECT_ROOT_DIR + "/" + config.get("RESULTS", "PLOTS_DIR")
RESULT_METRICS_DIR = PROJECT_ROOT_DIR + "/" + config.get("RESULTS", "METRICS_DIR")

PARAMETERS_FILE = PROJECT_ROOT_DIR + "/" + config.get("PARAMETERS", "PARAMS_PATH")
