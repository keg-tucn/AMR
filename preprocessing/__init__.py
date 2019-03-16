import ConfigParser

from definitions import CONFIG_FILE_PATH

config = ConfigParser.ConfigParser()
config.read(CONFIG_FILE_PATH)

STANFORD_NER_MODEL = config.get('STANFORD_NER', 'PARSER_MODEL')
STANFORD_NER_JAR = config.get('STANFORD_NER', 'PARSER_JAR')
