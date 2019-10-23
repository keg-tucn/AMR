class TrainData:

    def __init__(self, sentence, action_sequence, original_amr, dependencies, named_entities, date_entities,
                 concepts_metadata, amr_id):
        self.sentence = sentence
        self.action_sequence = action_sequence
        self.original_amr = original_amr
        self.dependencies = dependencies
        self.named_entities = named_entities
        self.date_entities = date_entities
        self.concepts_metadata = concepts_metadata
        self.amr_id = amr_id
