import json

from .keras_lstm_flow import *

if __name__ == "__main__":
    with open(PARAMETERS_FILE, "r") as f:
        parameters_dict = json.load(f)

        parser_parameters = ParserParameters.init_from_dict(parameters_dict)
