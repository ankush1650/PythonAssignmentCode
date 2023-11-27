import unittest
import pandas as pd
from src.function_engine import Function
from src.utilities.utils import OLSLoss
from unittest import TestCase


class Test(TestCase):
    def setUp(self):
        # Setting up some functions
        data1 = {"x": range(1, 10), "y": range(5, 14)}
        dataframe1 = pd.DataFrame(data=data1)

        data2 = {"x": range(1, 10), "y": range(7, 16)}
        dataframe2 = pd.DataFrame(data=data2)

        self.function1 = Function("func1")
        self.function1.dataframe = dataframe1

        self.function2 = Function("func2")
        self.function2.dataframe = dataframe2

    def test_ordinary_squared_error(self):
        # Test case 1: Ensure that the OLSLoss computes the correct error for different functions
        self.assertAlmostEqual(OLSLoss().ordinary_squared_error(self.function1, self.function2), 36.0)
        print("Testcase 1 passed")

        # Test case 2: Ensure that the OLSLoss returns 0 for identical functions
        self.assertAlmostEqual(OLSLoss().ordinary_squared_error(self.function1, self.function1), 0.0)
        print("Testcase 2 passed")

        # Test case 3: Ensure that the OLSLoss function works with empty functions
        empty_function = Function("empty")
        empty_function.dataframe = pd.DataFrame()
        self.assertEqual(OLSLoss().ordinary_squared_error(empty_function, empty_function), 0.0)
        print("Testcase 3 passed")


if __name__ == '__main__':
    unittest.main()
