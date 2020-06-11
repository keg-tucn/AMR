from data_extraction.dataset_reading_util import read_dataset_dict
from models.amr_graph import AMR


def get_no_nodes(amr: AMR, hist_concept: str, child_rel):
    no_nodes = 0
    for var,concept in amr.node_to_concepts.items():
        if concept == hist_concept:
            children_dict = amr[var]
            if child_rel in children_dict.keys():
                no_nodes += 1
    return no_nodes


def get_no_time_date_entities(amr: AMR):
    no_nodes = 0
    for var, concept in amr.node_to_concepts.items():
        if concept == 'date-entity':
            children_dict = amr[var]
            if 'time' in children_dict.keys():
                no_nodes+=1
    return no_nodes

def create_stanford_ner_preprocessing_histogram():
    """
        Create a histogram to see how many person, organization, percentage-entity and date-entity (with child rel time)
        are in the training dataset, to see which preprocessing is worth implementing
    """
    training_dataset_dict = read_dataset_dict('training')
    histogram = {'person': 0, 'organization': 0, 'percentage-entity':0,'time date-entity':0}
    for dataset, data_per_dataset in training_dataset_dict.items():
        for sentence, amr_str, amr_id in data_per_dataset:
            amr: AMR = AMR.parse_string(amr_str)
            histogram['person'] += get_no_nodes(amr,'person','name')
            histogram['organization'] += get_no_nodes(amr,'organization','name')
            histogram['percentage-entity'] += get_no_nodes(amr,'percentage-entity','value')
            histogram['time date-entity'] += get_no_nodes(amr,'date-entity','time')
    return histogram


if __name__ == "__main__":
    print(create_stanford_ner_preprocessing_histogram())

