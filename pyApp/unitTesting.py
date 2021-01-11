
import unittest
from arimamodel import getPredGraph, getProfits, getDataframe, getCSVS, getPred, trainModel
from functions import calcProfit

class TestStringMethods(unittest.TestCase):
    #---------- arimamodel.py Tests ----------

    def test_getCSVS(self):
        print("test_getCSVS Start")
        self.assertTrue(getCSVS())
        print("test_getCSVS Complete")

    def test_getPredGraph(self):
        print("test_getPredGraph Start")
        self.assertTrue(getPredGraph(2, "TSLA"))
        print("test_getPredGraph Complete")

    def test_getProfits(self):
        print("test_getProfits Start")
        self.assertIsNotNone(getProfits(7))
        print("test_getProfits Complete")

    def test_getDataframe(self):
        print("test_getDataframe Start")
        self.assertIsNotNone(getDataframe("TSLA"))
        print("test_getDataframe Complete")

    def test_getPred(self):
        print("test_getPred Start")
        pred = getPred(7, getDataframe("TSLA"), "TSLA")
        self.assertIsNotNone(pred)
        # len(pred) is 1 more than period of days
        # because it counts the current day and
        # 7 days after
        self.assertEqual(8, len(pred))
        print("test_getPred Complete")

    def test_trainModel(self):
        print("test_trainModel Start")
        self.assertIsNotNone(trainModel("TSLA"))
        print("test_trainModel Complete")
    #---------- functions.py Tests ----------

    def test_calcprofit(self):
        print("test_calcprofit Start")
        self.assertIsNotNone(calcProfit(50, 7))
        print("test_calcprofit Complete")

if __name__ == '__main__':
    unittest.main()