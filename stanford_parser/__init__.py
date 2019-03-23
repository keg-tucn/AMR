import ConfigParser

from definitions import CONFIG_FILE_PATH, PROJECT_ROOT_DIR

config = ConfigParser.ConfigParser()
config.read(CONFIG_FILE_PATH)

STANFORD_DEP_PARSER_HOME = PROJECT_ROOT_DIR + config.get('STANFORD_DEP_PARSER', 'PARSER_HOME')
