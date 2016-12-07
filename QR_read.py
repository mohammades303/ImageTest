__author__ = 'Mohammad'

import qrtools
import zbar
import numpy as np
import cv2
from PIL import Image

# This function reads and returns the QR code in input 'img'
# 'img' should be a cv2 image, i.e. read with cv2.imread(...)

def QR_read(img):

    scanner = zbar.ImageScanner()
    # configure the reader
    scanner.parse_config('enable')
    cv2_im_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    pil = pil_im.convert('L')
    width, height = pil.size
    raw = pil.tobytes()
    # wrap image data
    image = zbar.Image(width, height, 'Y800', raw)
    result = scanner.scan(image)
    if result:
        for symbol in image:
            return(symbol.data.decode(u'utf-8'))
    else:
        return(False)
        
        
################## Test ##############
# These can be used to test  QR_read() function
'''
A = cv2.imread('BVZ0072-QR_full_res.jpg')
print(QR_read(A))
A = cv2.imread('BVZ0073-QR_full_res.jpg')
print(QR_read(A))
A = cv2.imread('./QR_test_images/BVZ0073-QR-test.jpg')
print(QR_read(A))
A = cv2.imread('./QR_test_images/BVZ0057-QR-test.jpg')
print(QR_read(A))
A = cv2.imread('./QR_test_images/BVZ0069-QR-test.jpg')
print(QR_read(A))
'''