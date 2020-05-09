from smatch.smatch_amr import AMR


def test_parse_AMR_line():
    amr_str = """(y2 / year
                      :time-of (r / recover-01
                            :ARG1-of (e / expect-01 :polarity -))
                      :ARG1-of (p / possible-01)
                      :domain (d / date-entity :year 2012))"""
    amr = AMR.parse_AMR_line(amr_str)
    print(amr)


if __name__ == "__main__":
    test_parse_AMR_line()
    print("Everything in smatch_amr_test passed")
