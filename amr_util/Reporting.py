from smatch import smatch_util


class AMRResult:
    @staticmethod
    def headers():
        return "path,epoch,type,size,accuracy,invalid_actions,bins,histogram,%s" % smatch_util.AMRSmatchResult.headers()

    @staticmethod
    def histogram_beans():
        return [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1]

    def __init__(self, path, epoch, type, size, accuracy, invalid_actions, bins, histogram, smatch_result):
        self.path = path
        self.epoch = epoch
        self.type = type
        self.size = size
        self.accuracy = accuracy
        self.invalid_actions = invalid_actions
        self.bins = bins
        self.histogram = histogram
        self.smatch_result = smatch_result

    def __str__(self):
        return "%s,%d,%s,%d,%f,%d,%s,%s,%s" % (
            self.path, self.epoch, self.type, self.size, self.accuracy, self.invalid_actions, self.bins, self.histogram,
            self.smatch_result)
