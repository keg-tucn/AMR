import pandas as pd
import plotly.graph_objects as go


class DatasetAnalysis():

    def create_property_set(self, dataset_dictionary):
        """
        Takes as input a dataset dictionary:
            {dataset: [(sentence, custom_amr, amr_id)]
            eg. dataset = bolt
        Creates a set of a particular property for each dataset
            {dataset: [property]}
            eg. a set of all concepts in bolt, all concepts in deft etc.
                {bolt: {c1,c2,c3}, deft: {c4,c1,c5}}
        """
        raise NotImplemented()

    def get_counts(self, set1, set2):

        count1 = len(set1)
        count2 = len(set2)
        count_inters = len(set1.intersection(set2))
        count_only_set_1 = len(set1.difference(set2))
        count_only_set_2 = len(set2.difference(set1))
        return count1, count2, count_inters, count_only_set_1, count_only_set_2

    # TODO: total row (not enough to add values, must make new set)
    def create_comparative_analysis(self, path, property_name,
                                    dataset_dictionary1, dataset_dictionary2,
                                    set1_name, set2_name):
        """
        Takes as input 2 dataset dictionaries
            eg, datasets for training and dev by dataset name (bolt, defta, etc)
        Creates a table that counts how many properties are in:
            set 1,
            set 2,
            set 1 and set 2 (interesection),
            set 1 but not in set 2
            set 2 but not in set 1
        Usage example:
            To see:
                * how many concepts there are in training
                * how many concepts there are in dev
                * how many concepts there are that are found both in training and dev
                * how many concepts are only found in training
                * how many concepts are only found in dev
        """
        set1_dict = self.create_property_set(dataset_dictionary1)
        set2_dict = self.create_property_set(dataset_dictionary2)
        dataframe_data = []

        # make sure the sets dicts have the same keys
        for key1 in set1_dict.keys():
            if key1 not in set2_dict.keys():
                set2_dict[key1] = set()
        for key2 in set2_dict.keys():
            if key2 not in set1_dict.keys():
                set1_dict[key2] = set()

        total_set_1 = set()
        total_set_2 = set()
        for dataset, set1 in set1_dict.items():
            set2 = set2_dict[dataset]
            count1, count2, count_inters, count_only_set_1, count_only_set_2 = self.get_counts(set1, set2)
            total_set_1 = total_set_1.union(set1)
            total_set_2 = total_set_2.union(set2)
            dataframe_data.append([dataset, count1, count2, count_inters, count_only_set_1, count_only_set_2])
        # get total counts
        count1, count2, count_inters, count_only_set_1, count_only_set_2 = self.get_counts(total_set_1,
                                                                                           total_set_2)
        dataframe_data.append(['total', count1, count2, count_inters, count_only_set_1, count_only_set_2])

        cols = ['Datasets',
                '#' + property_name + ' in ' + set1_name,
                '#' + property_name + ' in ' + set2_name,
                '#' + property_name + ' in both' + set1_name + ' and ' + set2_name,
                '#' + property_name + ' only in ' + set1_name,
                '#' + property_name + ' only in ' + set2_name
                ]
        df = pd.DataFrame(dataframe_data, columns=cols)
        df.to_excel(path, index=False, header=True)
