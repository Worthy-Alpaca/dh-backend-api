import unittest
import os
import sys

PACKAGE_PARENT = "../"
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from pathlib import Path
from src.data.dataloader import DataLoader


class TestDataLoader(unittest.TestCase):
    def test_return_value(self):
        """Test if DataLoader returns the correct dtype"""
        path = Path(os.getcwd() + os.path.normpath("/data/programms/24aarab/m10"))
        result = DataLoader(path)()
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()
