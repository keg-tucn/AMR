from AMRGraph import AMR
from AMRData import CustomizedAMR

amr = AMR.parse_string('(l / look-01~e.3,5 '
      ':ARG0 (i / i~e.0) '
      ':mod (j / just~e.2))')
camr=CustomizedAMR()
camr.createCustomAMR(amr)

