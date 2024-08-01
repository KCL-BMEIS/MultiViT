import unittest

import numpy as np

import torch

from multivit.modules import MappingGenerator, MultiViTPatchMapper

class TestMappingGenerator(unittest.TestCase):

    def test_mapping_single_image_multiplier_1(self):
        mg = MappingGenerator(16, 1, 1)

        # interleave increasing values with decreasing values (0, 63, 1, 62...)
        with torch.no_grad():
            importances = torch.stack((torch.arange(0, 32), torch.arange(63, 31, -1)), dim=1).flatten()
            actual = mg([importances])
            print(actual)

    def test_mapping_single_image_multiplier_2(self):
        mg = MappingGenerator(16, 1, 2)

        # interleave increasing values with decreasing values (0, 63, 1, 62...)
        with torch.no_grad():
            importances = torch.stack((torch.arange(0, 32), torch.arange(63, 31, -1)), dim=1).flatten()
            importances = importances[None, :]
            print(importances.shape)
            actual = mg([importances])
            print(actual)

    def test_mapping_multi_image_multiplier_2(self):
        mg = MappingGenerator(8, 8, 16, 2, 2)

        with torch.no_grad():
            image_0 = np.zeros(64)
            image_0[2] = 16
            image_0[10:13] = [15, 13, 14]
            image_0[13] = 8
            image_0[63] = 7
            image_0[37] = 4
            image_0[29] = 1

            image_1 = np.zeros(64)
            image_1[31] = 12
            image_1[5] = 11
            image_1[13] = 10
            image_1[14] = 9
            image_1[42:44] = [5, 6]
            image_1[50:52] = [2, 3]
            importances = [
                torch.tensor(image_0)[None, :],
                torch.tensor(image_1)[None, :]
            ]
            expected = [
                [
                    [(0, 2), [0, 1, 2, 3]],
                    [(0, 10), [4, 5, 6, 7]],
                    [(0, 12), [8, 9, 10, 11]],
                    [(0, 11), [12, 13, 14, 15]],
                    [(1, 31), [16, 17, 18, 19]],
                    [(1, 5), [20, 21, 22, 23]],
                    [(1, 13), [24, 25, 26, 27]],
                    [(1, 14), [28, 29, 30, 31]],
                    [(0, 13), [32, 33, 34, 35]],
                    [(0, 63), [36, 37, 38, 39]],
                    [(1, 43), [40, 41, 42, 43]],
                    [(1, 42), [44, 45, 46, 47]],
                    [(0, 37), [48, 49, 50, 51]],
                    [(1, 51), [52, 53, 54, 55]],
                    [(1, 50), [56, 57, 58, 59]],
                    [(0, 29), [60, 61, 62, 63]],
                ]
            ]

            expected_hr = [
                [
                    [(0, 4), [0]], [(0, 5), [1]], [(0, 20), [2]], [(0, 21), [3]],
                    [(0, 36), [4]], [(0, 37), [5]], [(0, 52), [6]], [(0, 53), [7]],
                    [(0, 40), [8]], [(0, 41), [9]], [(0, 56), [10]], [(0, 57), [11]],
                    [(0, 38), [12]], [(0, 39), [13]], [(0, 54), [14]], [(0, 55), [15]],
                    [(1, 110), [16]], [(1, 111), [17]], [(1, 126), [18]], [(1, 127), [19]],
                    [(1, 10), [20]], [(1, 11), [21]], [(1, 26), [22]], [(1, 27), [23]],
                    [(1, 42), [24]], [(1, 43), [25]], [(1, 58), [26]], [(1, 59), [27]],
                    [(1, 44), [28]], [(1, 45), [29]], [(1, 60), [30]], [(1, 61), [31]],
                    [(0, 42), [32]], [(0, 43), [33]], [(0, 58), [34]], [(0, 59), [35]],
                    [(0, 238), [36]], [(0, 239), [37]], [(0, 254), [38]], [(0, 255), [39]],
                    [(1, 166), [40]], [(1, 167), [41]], [(1, 182), [42]], [(1, 183), [43]],
                    [(1, 164), [44]], [(1, 165), [45]], [(1, 180), [46]], [(1, 181), [47]],
                    [(0, 138), [48]], [(0, 139), [49]], [(0, 154), [50]], [(0, 155), [51]],
                    [(1, 198), [52]], [(1, 199), [53]], [(1, 214), [54]], [(1, 215), [55]],
                    [(1, 196), [56]], [(1, 197), [57]], [(1, 212), [58]], [(1, 213), [59]],
                    [(0, 106), [60]], [(0, 107), [61]], [(0, 122), [62]], [(0, 123), [63]],
                ]
            ]
            actual, actual_hr = mg(importances)
            self.assertEqual(len(actual), 1)
            self.assertEqual(len(actual[0]), len(expected[0]))
            self.assertEqual(len(actual_hr), 1)
            self.assertEqual(len(actual_hr[0]), len(expected_hr[0]))
            for i in range(len(expected[0])):
                self.assertTupleEqual(actual[0][i][0], expected[0][i][0])
                self.assertListEqual(actual[0][i][1], expected[0][i][1])
            for i in range(len(expected_hr[0])):
                self.assertTupleEqual(actual_hr[0][i][0], actual_hr[0][i][0])
                self.assertListEqual(actual_hr[0][i][1], expected_hr[0][i][1])



    def test_map_entries_to_high_res(self):

        src = [
                [
                    [(0, 2), [0, 1, 2, 3]],
                    [(0, 10), [4, 5, 6, 7]],
                    [(0, 12), [8, 9, 10, 11]],
                    [(0, 11), [12, 13, 14, 15]],
                    [(1, 31), [16, 17, 18, 19]],
                    [(1, 5), [20, 21, 22, 23]],
                    [(1, 13), [24, 25, 26, 27]],
                    [(1, 14), [28, 29, 30, 31]],
                    [(0, 13), [32, 33, 34, 35]],
                    [(0, 63), [36, 37, 38, 39]],
                    [(1, 43), [40, 41, 42, 43]],
                    [(1, 42), [44, 45, 46, 47]],
                    [(0, 37), [48, 49, 50, 51]],
                    [(1, 51), [52, 53, 54, 55]],
                    [(1, 50), [56, 57, 58, 59]],
                    [(0, 29), [60, 61, 62, 63]],
                ]
            ]

        actual = MappingGenerator(8, 8, 64, 2, 2).map_entries_to_high_res(src)
        for b in actual:
            for a in b:
                print(a)



class TestMultiViTPatchMapper(unittest.TestCase):


    def test_mapping_multiplier_1(self):
        images = torch.zeros((1, 2, 3, 16, 16))
        ps = 4
        print()
        for i, (z, y, x) in enumerate(np.ndindex((2, 4, 4))):
            images[0, z, :, y*ps:(y+1)*ps, x*ps:(x+1)*ps] = i

        mapping = [
                    [
                        [(0, 2), [0]],
                        [(0, 9), [1]],
                        [(1, 0), [2]],
                        [(1, 15), [3]],
                        [(0, 7), [4]],
                        [(1, 2), [5]],
                        [(1, 9), [6]],
                        [(0, 4), [7]],
                        [(0, 8), [8]],
                        [(1, 7), [9]],
                        [(0, 13), [10]],
                        [(1, 5), [11]],
                        [(1, 6), [12]],
                        [(0, 14), [13]],
                        [(1, 3), [14]],
                        [(0, 1), [15]],
                    ]
                  ]

        mapper = MultiViTPatchMapper(4, 4, 4, 1)
        actual = mapper(images, mapping)

        for i, (y, x) in enumerate(np.ndindex((4, 4))):
            expected = mapping[0][i][0][0] * 16 + mapping[0][i][0][1]
            self.assertTrue(
                torch.all(actual[0, :, y*ps:(y+1)*ps, x*ps:(x+1)*ps] == expected)
            )
