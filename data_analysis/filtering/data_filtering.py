from models.amr_data import CustomizedAMR


class CustomizedAMRFilterParams:

    def __init__(self, sentence, custom_amr: CustomizedAMR, amr_id):
        self.sentence = sentence
        self.custom_amr = custom_amr
        self.amr_id = amr_id


class DataFiltering:
    """
    data: given as a list of (sentence,amr) pairs
    """

    def __init__(self, data):
        self.filters = []
        self.data = data

    def add_filter(self, filter):
        self.filters.append(filter)

    def create_filter_params(self, data_entry):
        raise NotImplemented()

    def execute(self):
        new_data = []
        for (data_entry) in self.data:

            pass_filters = True

            for f in self.filters:
                filter_params = self.create_filter_params(data_entry)
                if not f.is_ok(filter_params):
                    pass_filters = False
                    break

            if pass_filters:
                new_data.append(data_entry)
        return new_data


class CustomizedAMRDataFiltering(DataFiltering):
    """
    For when the data to be filtered looks like: sentence, custom_amr, amr_id
    Associated filters can be found in filters.py
    """

    def __init__(self, data):
        super().__init__(data)

    def create_filter_params(self, data_entry):
        return CustomizedAMRFilterParams(data_entry[0], data_entry[1], data_entry[2])
