class PreprocessingMetadata:

    def __init__(self):
        self.preprocessing_steps = []
        self.preprocessing_metadata = {}

    def add_step(self, step_name: str, step_metadata):
        self.preprocessing_steps.append(step_name)
        self.preprocessing_metadata[step_name] = step_metadata
