import unittest
import matplotlib.pyplot as plt
import cv2
import struct
import tifffile
import numpy as np
import matplotlib.pyplot as plt



class testBMPLoad(unittest.TestCase):
    def test_BMP(self):

        img = cv2.imread("tests/data/scgmair3008.007.bmp",cv2.IMREAD_UNCHANGED)
        plt.imshow(img[:, :, 0], cmap='gray')
        plt.colorbar()
        plt.show()
        plt.savefig("bmp.png")
        self.assertTrue((img[:, :, 0] == img[:, :, 1]).all())  # All channel the same


if __name__ == '__main__':
    unittest.main()
