import unittest
from preprocessing import SentenceAMRPairsExtractor, ActionSequenceGenerator

class TestBacktrackingASG(unittest.TestCase):

    def setUp(self):
        file_path = 'resources/alignments/split/training/deft-p2-amr-r1-alignments-training-bolt.txt'
        sentence_amr_triples = SentenceAMRPairsExtractor.extract_sentence_amr_pairs(file_path)


    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()