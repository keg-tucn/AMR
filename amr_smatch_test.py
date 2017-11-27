'''
Checks that printing a parsed AMR yields the same thing.
'''

from os import listdir

import TrainingDataExtractor as tde
from postprocessing import ActionSequenceReconstruction as asr
from smatch import smatch_amr
from smatch import smatch_util


def check_smatch_identical(print_info, amrstr1, amrstr2):
    amr2 = smatch_amr.AMR.parse_AMR_line(amrstr2)
    if amr2 is None:
        print (print_info + " Could not reparse!!!")
        print("Original:")
        print amrstr1
        print("Could not reparse:")
        print amrstr2
        amr2 = smatch_amr.AMR.parse_AMR_line(amrstr2)
        return False

    else:
        smatch_f_score = smatch_util.smatch_f_score(amr1, amr2)
        if smatch_f_score < 1:
            print(print_info + " Score %f" % smatch_f_score)
            print("Original:")
            print amrstr1
            print("Reconstructed:")
            print amrstr2
            # amr1.pretty_print()
            smatch_amr.AMR.parse_AMR_line(amr_string)
            return False

    return True


data = []
mypath = 'resources/alignments/split/dev'
directory_content = listdir(mypath)
original_corpus = filter(lambda x: "dump" not in x and "audit" not in x, directory_content)
for f in original_corpus:
    mypath_f = mypath + "/" + f
    print(mypath_f)
    data += tde.generate_training_data(mypath_f).data

fail_count = 0
for elem in data:
    sentence = elem.sentence
    action_sequence = elem.action_sequence
    amr_string = elem.amr_original
    amr1 = smatch_amr.AMR.parse_AMR_line(amr_string)
    if amr1 is None:
        print 'Could not parse original amr'
        print amr_string
        fail_count += 1

    reprinted = amr1.pretty_print()
    ok = check_smatch_identical('Reparse.', amr_string, reprinted)

    rec_string = asr.reconstruct_all(action_sequence)
    amr1 = smatch_amr.AMR.parse_AMR_line(amr_string)
    ok = ok and check_smatch_identical('Reconstruct from actions.', amr_string, rec_string)

    if not ok:
        fail_count += 1

print("FAIL COUNT: %i / %i" % (fail_count, len(data)))
