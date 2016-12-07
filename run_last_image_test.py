##############################################################################################
#
#       This is a function developed for analyzing top view plant images that contains 
#       a color card and some QR codes. An annotated image along with a number of 
#       information and stats are generated.
#
#       Mohammad Esmaeilzadeh, Borevitz lab, Australian National University
#       mohammad.esmaeilzadeh@anu.edu.au, m.esmaielzadeh@gmail.com           
#
##############################################################################################


import numpy as np
import cv2
from imutils import cord_rotate
from Detect_colorChecker import detect_card
from Color_correction import Color_correct_stats
from QR_read import QR_read


# CONSTANTS
COLOR_THRESHOLD = 120
SCALE_SEARCH_RANGE = np.linspace(0.9,1.1,11)
DEGREE_SEARCH_RANGE = np.linspace(-5,5,21)


def run_test(img_file, card_file, mask_file, QR_file, NUM_cores):
    
    Output = {  'Card_Detection_Accuracy' : 100,
                'Card_Degree' : 0,
                'Card_Orientation' : True,
                'Card_Damaged/blocked' : False,
                'Color_Correction_Error' : 0,
                'NO_QR_detected' : 8,
                'NO_QR_readable' : 8,
                'Average_QR_degree' : 0,
                'Median_QR_degree' : 0,
                'Min_QR_detection_Accuracy' : 100,
                'std_QR_cord_top_row' : 0,
                'std_QR_cord_bottom_row' : 0,
                'R_sum_percentage' : 33.33,
                'G_sum_percentage' : 33.33,
                'B_sum_percentage' : 33.33
              }
              
    # Reading input image
    img = cv2.imread(img_file)
    output_img = img
    img_centre = [img.shape[1]/2,img.shape[0]/2]
    rect_margin = int(10*float(img.shape[0])/720)
    rect_thick = int(2*float(img.shape[0])/720)

    # Caclulating the ratio of R,G,B pixels and filling the relevant output fields
    b,g,r = cv2.split(img)
    Rsum =  np.sum(r>COLOR_THRESHOLD)
    Gsum =  np.sum(g>COLOR_THRESHOLD)
    Bsum =  np.sum(b>COLOR_THRESHOLD)
    TotSum = Rsum + Gsum + Bsum
    Output['R_sum_percentage'] = round(10000*Rsum/float(TotSum))/float(100)
    Output['G_sum_percentage'] = round(10000*Gsum/float(TotSum))/float(100)
    Output['B_sum_percentage'] = round(10000*Bsum/float(TotSum))/float(100)
    
    # Reading an binarizing and analyzing the mask for detecting QR codes
    mask = cv2.imread(mask_file,0)
    ret, mask = cv2.threshold(mask, 10, 1, cv2.THRESH_BINARY)
    mask_analyzed = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    
    # Reading card template and detecting the edges
    card = cv2.imread(card_file,0)
    card = cv2.Canny(card, 40, 50)
    
    # Reading QR template and detecting the edges
    qr = cv2.imread(QR_file,0)
    qr = cv2.Canny(qr, 40, 50)
    
    # Cropping Image to area where color card is supposed to be
    crop_s_x = img.shape[1]/4
    crop_e_x = 3*img.shape[1]/4
    crop_s_y = img.shape[0]/4
    crop_e_y = 3*img.shape[0]/4
    img_cropped = img[crop_s_y:crop_e_y,crop_s_x:crop_e_x,:]
    img_cropped_center = [img_cropped.shape[1]/2,img_cropped.shape[0]/2]
    
    # Detecting color card and filling the relevant output fields 
    Detected_card, Acc, scale, deg, TotSum, startX, startY, endX, endY  = detect_card(img_cropped,card,SCALE_SEARCH_RANGE, DEGREE_SEARCH_RANGE, NUM_cores)
    Output['Card_Detection_Accuracy'] = round(10000*Acc)/float(100)
    Output['Card_Degree'] = deg
    
    # Mapping back X and Y coordinates of detected card to the original image coordinates
    startX, startY = cord_rotate(img_cropped_center, [startX, startY], deg*3.1415/float(180))
    startX = startX + crop_s_x
    startY = startY + crop_s_y
    endX, endY = cord_rotate(img_cropped_center, [endX, endY], deg*3.1415/float(180))
    endX = endX + crop_s_x
    endY = endY + crop_s_y
    
    # Checking card and filling the relevant output fields
    Output['Card_Orientation'], Output['Card_Damaged/blocked'], Output['Color_Correction_Error'] = Color_correct_stats(Detected_card, Acc)
    
    # Drawing a box around the detected card: green if it is all good, red otherwise
    if Output['Card_Damaged/blocked'] or not Output['Card_Orientation'] or Output['Card_Detection_Accuracy']<30:
        cv2.rectangle(output_img,(startX-rect_margin,startY-rect_margin),(endX+rect_margin,endY+rect_margin),(0,0,255),rect_thick)
    else:
        cv2.rectangle(output_img,(startX-rect_margin,startY-rect_margin),(endX+rect_margin,endY+rect_margin),(0,255,0),rect_thick)
    
    ########################################## Analyzing QR ##############################################
    # This whole section deals with detecting and reading QR codes based on the QR template and the mask
    # This whole section can be safely commented if stats for the QR codes is not of interest
    
    X = np.zeros(8)
    Y = np.zeros(8)
    TotSum = np.zeros(8)
    deg = np.zeros(8)
    Acc = np.zeros(8)
    QR_readable_cnt = 0
    
    for i in [1,2,3,4,5,6,7,8]: # Analayzing the 8 spots where QR codes are supposed to be
         
        # Cropping image to location of each indivdual QR code
        mask_temp = np.where(mask_analyzed[1]==i)
        crop_s_x = min(mask_temp[1])
        crop_e_x = max(mask_temp[1])
        crop_s_y = min(mask_temp[0])
        crop_e_y = max(mask_temp[0])
        image_masked = img[crop_s_y:crop_e_y,crop_s_x:crop_e_x,:]
        img_masked_center = [image_masked.shape[1]/2,image_masked.shape[0]/2]

        # Detecting QR code
        Detected_QR, Acc[i-1], scale, deg[i-1], TotSum[i-1], startX, startY,endX, endY = detect_card(image_masked,qr,SCALE_SEARCH_RANGE, DEGREE_SEARCH_RANGE, NUM_cores)
        
        # Reading QR code
        decoded_QR = QR_read(Detected_QR)
        
        # Counting number of QR that can be read
        if decoded_QR is not False:
            QR_readable_cnt = QR_readable_cnt + 1
            Output['QR_readable_'+str(QR_readable_cnt)] = str(decoded_QR)
        
        # Mapping back X and Y coordinates of detected card to the original image coordinates
        startX ,startY = cord_rotate(img_masked_center, [startX, startY], deg[i-1]*3.1415/float(180))
        startX = startX + crop_s_x
        startY = startY + crop_s_y
        endX ,endY = cord_rotate(img_masked_center, [endX, endY], deg[i-1]*3.1415/float(180))
        endX = endX + crop_s_x
        endY = endY + crop_s_y
        X[i-1] = startX
        Y[i-1] = startY

        # Drawing a box around the detected QR code: green if could be detected and read, blue if detected but not read, red otherwise
        if TotSum[i-1] < 2.5 or Acc[i-1] < 0.1:
            cv2.rectangle(output_img,(startX-rect_margin,startY-rect_margin),(endX+rect_margin,endY+rect_margin),(0,0,255),rect_thick)
        elif decoded_QR is False: 
            cv2.rectangle(output_img,(startX-rect_margin,startY-rect_margin),(endX+rect_margin,endY+rect_margin),(255,0,0),rect_thick)
        else:
            cv2.rectangle(output_img,(startX-rect_margin,startY-rect_margin),(endX+rect_margin,endY+rect_margin),(0,255,0),rect_thick)
    
    # Filling the the rest of output fields
    Output['NO_QR_detected'] = np.sum(TotSum > 2.5)
    Output['NO_QR_readable'] = QR_readable_cnt
    Output['Average_QR_degree'] = np.mean(deg)
    Output['Median_QR_degree'] = np.median(deg)
    Output['Min_QR_detection_Accuracy'] = round(10000*np.min(Acc))/float(100)
    Output['std_QR_cord_top_row'] = int(np.std(Y[0:4])) # to check if the top row QR codes are in one line
    Output['std_QR_cord_bottom_row'] = int(np.std(Y[4:8])) # to check if the bottom row QR codes are in one line
    
    ####################################### End of Analyzing QR #########################################
    
    return(output_img,Output)
    

    
    
    
#################################### Run the Test #####################################

# Select one of the test scenarios and run:
# output_img,Output = run_test(INPUT_FILE, CARD, MASK, QR, 8)


'''
#### Test scenario 1 ##########
CARD = './Templates and Masks/card.jpg'
MASK = './Templates and Masks/mask.jpg'
INPUT_FILE = './Test_images/Low_res/BVZ0039-GC03R-C01~fullres-orig_2015_05_15_15_35_00_00.jpg'
QR = './Templates and Masks/BVZ0039-QR.jpg'
###############################
'''

'''
#### Test scenario 2 ##########
CARD = './Templates and Masks/card.jpg'
MASK = './Templates and Masks/mask_rotated.jpg'
INPUT_FILE = './Test_images/Low_res/BVZ0039-GC03L-C01~fullres-orig_2015_05_14_12_20_00_00.jpg'
QR = './Templates and Masks/BVZ0039-QR_rotated.jpg'
###############################
'''

'''
#### Test scenario 3 ##########
MASK = './Templates and Masks/mask.jpg'
CARD = './Templates and Masks/card.jpg'
INPUT_FILE = './Test_images/Low_res/BVZ0063-GC02R-C01~fullres-orig_2016_04_13_12_30_00_00.jpg'
QR = './Templates and Masks/BVZ0063-QR.jpg'
###############################
'''

'''
#### Test scenario 4 ##########
MASK = './Templates and Masks/mask.jpg'
CARD = './Templates and Masks/card.jpg'
INPUT_FILE = './Test_images/Low_res/BVZ0071-GC36R-RGB01~fullres-orig_2016_09_25_13_25_00_00.jpg'
QR = './Templates and Masks/BVZ0071-QR.jpg'
###############################
'''

'''
#### Test scenario 5 ##########
MASK = './Templates and Masks/mask.jpg'
CARD = './Templates and Masks/card.jpg'
INPUT_FILE = './Test_images/Low_res/BVZ0072-GC35L-RGB01~fullres-orig_2016_09_29_12_20_00_00.jpg'
QR = './Templates and Masks/BVZ0072-QR.jpg'
###############################
'''

'''
#### Test scenario 6 ##########
MASK = './Templates and Masks/mask.jpg'
CARD = './Templates and Masks/card.jpg'
INPUT_FILE = './Test_images/Low_res/BVZ0072-GC35L-RGB01~fullres-orig_2016_09_30_13_50_00_00.jpg'
QR = './Templates and Masks/BVZ0072-QR.jpg'
###############################
'''

'''
#### Test scenario 7 ##########
MASK = './Templates and Masks/mask.jpg'
CARD = './Templates and Masks/card.jpg'
INPUT_FILE = './Test_images/Low_res/BVZ0072-GC35L-RGB01~fullres-orig_2016_10_21_14_30_00_00.jpg'
QR = './Templates and Masks/BVZ0072-QR.jpg'
###############################
'''

'''
#### Test scenario 8 ##########
MASK = './Templates and Masks/mask.jpg'
CARD = './Templates and Masks/card.jpg'
INPUT_FILE = './Test_images/Low_res/BVZ0072-GC35R-RGB01~fullres-orig_2016_09_30_13_45_00_00.jpg'
QR = './Templates and Masks/BVZ0072-QR.jpg'
###############################
'''

'''
#### Test scenario 9 ##########
MASK = './Templates and Masks/mask.jpg'
CARD = './Templates and Masks/card.jpg'
INPUT_FILE = './Test_images/Low_res/BVZ0073-GC36L-RGB01~fullres-orig_2016_09_29_11_25_00_00.jpg'
QR = './Templates and Masks/BVZ0071-QR.jpg' # the QR for BVZ0071 is used since they are not updated in this image yet!
###############################
'''

'''
#### Test scenario 10 ##########
MASK = './Templates and Masks/mask.jpg'
CARD = './Templates and Masks/card.jpg'
INPUT_FILE = './Test_images/Low_res/BVZ0073-GC36R-RGB01~fullres-orig_2016_11_05_11_25_00_00.jpg'
QR = './Templates and Masks/BVZ0073-QR.jpg'
###############################
'''

'''
#### Test scenario 11 ##########
MASK = './Templates and Masks/mask_full_res.jpg'
INPUT_FILE = './Test_images/Full_res/BVZ0072-GC35L-RGB01~fullres-orig_2016_11_07_14_30_00_00.jpg'
QR = './Templates and Masks/BVZ0072-QR_full_res.jpg'
CARD = './Templates and Masks/card_full_res.jpg'
###############################
'''

'''
#### Test scenario 12 ##########
MASK = './Templates and Masks/mask_full_res.jpg'
INPUT_FILE = './Test_images/Full_res/BVZ0072-GC35R-RGB01~fullres-orig_2016_09_30_13_45_00_00.jpg'
QR = './Templates and Masks/BVZ0072-QR_full_res.jpg'
CARD = './Templates and Masks/card_full_res.jpg'
###############################
'''

'''
#### Test scenario 13 ##########
MASK = './Templates and Masks/mask_full_res.jpg'
INPUT_FILE = './Test_images/Full_res/BVZ0073-GC36R-RGB01~fullres-orig_2016_11_05_11_25_00_00.jpg'
QR = './Templates and Masks/BVZ0072-QR_full_res.jpg'
CARD = './Templates and Masks/card_full_res.jpg'
###############################
'''

'''
#### Test scenario 14 ##########
MASK = './Templates and Masks/mask_full_res.jpg'
INPUT_FILE = './Test_images/Full_res/BVZ0072-GC35R-RGB01~fullres-orig_2016_10_24_14_35_00_00.jpg'
QR = './Templates and Masks/BVZ0072-QR_full_res.jpg'
CARD = './Templates and Masks/card_full_res.jpg'
###############################
'''

'''
#### Test scenario 15 ##########
MASK = './Templates and Masks/mask_full_res.jpg'
INPUT_FILE = './Test_images/Full_res/BVZ0072-GC35L-RGB01~fullres-orig_2016_09_30_13_50_00_00.jpg'
QR = './Templates and Masks/BVZ0072-QR_full_res.jpg'
CARD = './Templates and Masks/card_full_res.jpg'
###############################
'''

'''
#### Test scenario 16 ##########
MASK = './Templates and Masks/mask_full_res.jpg'
INPUT_FILE = './Test_images/Full_res/BVZ0073-GC36L-RGB01~fullres-orig_2016_10_27_12_30_00_00.jpg'
QR = './Templates and Masks/BVZ0072-QR_full_res.jpg'
CARD = './Templates and Masks/card_full_res.jpg'
###############################
'''


#### Test scenario 17 ##########
MASK = './Templates and Masks/mask_full_res.jpg'
INPUT_FILE = './Test_images/Full_res/BVZ0069-GC37L-C01~fullres-orig_2016_06_14_13_35_00_00.jpg'
QR = './Templates and Masks/BVZ0069-QR_full_res.jpg'
CARD = './Templates and Masks/card_full_res.jpg'
###############################


output_img,Output = run_test(INPUT_FILE, CARD, MASK, QR, 8)
print(Output)
cv2.imwrite(INPUT_FILE.replace('Test_images','Output'),output_img)


    


