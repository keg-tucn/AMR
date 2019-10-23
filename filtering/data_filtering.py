class DataFiltering:
    """
    data: given as a list of (sentence,amr) pairs
    """

    def __init__(self, data):
        self.filters = []
        self.data = data

    def add_filter(self, filter):
        self.filters.append(filter)

    def execute(self):
        new_data = []
        for (sentence, amr, custom_amr, amr_id) in self.data:

            pass_filters = True

            for f in self.filters:

                if not f.is_ok(sentence, amr, custom_amr, amr_id):
                    pass_filters = False
                    break

            if pass_filters:
                new_data.append((sentence, amr, custom_amr, amr_id))
        return new_data
