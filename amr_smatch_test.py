'''
Checks that printing a parsed AMR yields the same thing.
'''

from os import listdir, path
import TrainingDataExtractor as tde
from smatch import smatch_amr
from smatch import smatch_util


data = []
mypath = 'resources/alignments/split/dev'
directory_content = listdir(mypath)
original_corpus = filter(lambda x: "dump" not in x, directory_content)
for f in original_corpus:
    mypath_f = mypath + "/" + f
    print(mypath_f)
    data += tde.generate_training_data(mypath_f, False)

fail_count = 0
for (sentence, action_sequence, amr_string) in data:
    amr1 = smatch_amr.AMR.parse_AMR_line(amr_string)
    pretty = amr1.pretty_print()
    amr2 = smatch_amr.AMR.parse_AMR_line(pretty) # reparse
    if amr2 is None:
        print ("Could not reparse!!!")
        print("Original:")
        print amr_string
        print("Parsed, printed:")
        print pretty
        fail_count += 1

    else:
        smatch_f_score = smatch_util.smatch_f_score(amr1, amr2)
        if smatch_f_score < 1:
            print("Score %f" % smatch_f_score)
            print("Original:")
            print amr_string
            print("Parsed, printed:")
            print pretty
            #amr1.pretty_print()
            smatch_amr.AMR.parse_AMR_line(amr_string)
            fail_count += 1

print("FAIL COUNT: %i" % fail_count)
