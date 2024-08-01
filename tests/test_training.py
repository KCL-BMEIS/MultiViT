import unittest

from multivit import modules


class TestParseAdaptorArguments(unittest.TestCase):

    def _test_tuple_and_list(self, args, expected=None):
        actual = modules.parse_adaptor_arguments(args, 128, 1)
        if expected is None:
            raise ValueError("Shouldn't be able to reach here with expected = None")
        self.assertEqual(actual, expected)
        actual = modules.parse_adaptor_arguments(tuple(args), 128, 1)
        self.assertEqual(actual, expected)

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            self._test_tuple_and_list([])

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            self._test_tuple_and_list("mlp S128")

    def test_probe_with_no_arguments(self):
        self._test_tuple_and_list(
            ['probe'],
            {
                'adaptor': 'probe',
                'input_size': 128,
                'output_size': 1,
            }
        )

    def test_probe_with_additional_arguments(self):
        with self.assertRaises(ValueError):
            self._test_tuple_and_list(['probe', 'S128'], None)

    def test_mlp_with_no_additional_arguments(self):
        self._test_tuple_and_list(
            ['mlp'],
            {
                'adaptor': 'mlp',
                'input_size': 128,
                'hidden_size': 64,
                'hidden_layers': 1,
                'output_size': 1,
                'dropout': 0
            }
        )

    def test_mlp_with_one_argument(self):
        self._test_tuple_and_list(
            ['mlp', 'S256'],
            {
                'adaptor': 'mlp',
                'input_size': 128,
                'hidden_size': 256,
                'hidden_layers': 1,
                'output_size': 1,
                'dropout': 0
            }
        )

    def test_mlp_with_multiple_arguments(self):
        self._test_tuple_and_list(
            ['mlp', 'S128', 'L2', 'D10'],
            {
                'adaptor': 'mlp',
                'input_size': 128,
                'hidden_size': 128,
                'hidden_layers': 2,
                'output_size': 1,
                'dropout': 10
            }
        )

    def test_mlp_with_duplicate_arguments(self):
        with self.assertRaises(ValueError):
            self._test_tuple_and_list(
                ['mlp', 'S128', 'S256']
            )

    def test_mlp_with_unrecognized_argument(self):
        with self.assertRaises(ValueError):
            self._test_tuple_and_list(['mlp', 'X128'])

    def test_mlp_with_too_many_arguments(self):
        with self.assertRaises(ValueError):
            self._test_tuple_and_list(['mlp', 'S128', 'L2', 'D10', 'X1'])

# Add this to allow running the tests if this script is executed directly
if __name__ == '__main__':
    unittest.main()