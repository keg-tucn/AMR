import ConfigParser

from definitions import CONFIG_FILE_PATH

config = ConfigParser.ConfigParser()
config.read(CONFIG_FILE_PATH)

STANFORD_DEP_PARSER_HOME = config.get('STANFORD_DEP_PARSER', 'PARSER_HOME')
