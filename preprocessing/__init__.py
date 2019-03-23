import ConfigParser

from definitions import CONFIG_FILE_PATH, PROJECT_ROOT_DIR

config = ConfigParser.ConfigParser()
config.read(CONFIG_FILE_PATH)

STANFORD_NER_MODEL = PROJECT_ROOT_DIR + config.get('STANFORD_NER', 'PARSER_MODEL')
STANFORD_NER_JAR = PROJECT_ROOT_DIR + config.get('STANFORD_NER', 'PARSER_JAR')
