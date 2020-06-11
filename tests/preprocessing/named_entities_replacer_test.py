import unittest

from preprocessing.NamedEntitiesReplacer import process_sentence, process_language


class MyTestCase(unittest.TestCase):

    def test_process_sentence(self):
        generated_output = process_sentence('Rami Eid John is studying in San Francisco')
        expected_output = ('PERSON is studying in LOCATION ',
                           [(0, ['Rami', 'Eid', 'John']), (4, ['San', 'Francisco'])])
        self.assertEqual(generated_output, expected_output)

    def test_process_sentence_ex_2(self):
        generated_output = process_sentence('The center will bolster NATO \'s defenses against cyber attacks .')
        print(generated_output)
        expected_output = ('The center will bolster ORGANIZATION \'s defenses against cyber attacks . ',
                           [(4, ['NATO'])])
        self.assertEqual(generated_output, expected_output)


if __name__ == '__main__':
    unittest.main()
