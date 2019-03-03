from AMRGraph import AMR
import AMRData
from smatch import smatch_util
from smatch import smatch_amr


# method that returns the customized AMR representation directly from amr_str
def get_custom_amr(amr_str):
    amr = AMR.parse_string(amr_str)
    custom_amr = AMRData.CustomizedAMR()
    custom_amr.create_custom_AMR(amr)
    return custom_amr


# calculates the smatch of two amrs in string formar
def get_smatch(amr_str_1, amr_str_2):
    smatch_results = smatch_util.SmatchAccumulator()
    amr1 = smatch_amr.AMR.parse_AMR_line(amr_str_1)
    amr2 = smatch_amr.AMR.parse_AMR_line(amr_str_2)
    smatch_f_score = smatch_results.compute_and_add(amr1, amr2)
    return smatch_f_score
