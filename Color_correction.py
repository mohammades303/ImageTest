__author__ = 'Mohammad'

import numpy as np
import cv2
import colorbalance

def Color_correct_stats(card, Acc):
    CardRGB = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
    actual_colors, actual_colors_std = colorbalance.get_colorcard_colors(CardRGB,grid_size=[6, 4])
    cnt_color = 0
    card_orientation = True
    card_damaged = False
    if np.sum(actual_colors[:, 8])> np.sum(actual_colors[:, -9]):
        cnt_color = cnt_color + 1
    if np.sum(actual_colors[:, 5])> np.sum(actual_colors[:, -6]):
        cnt_color = cnt_color + 1
    if np.sum(actual_colors[:, 0])< np.sum(actual_colors[:, -1]):
        cnt_color = cnt_color + 1
    if cnt_color >= 2:
        actual_colors = actual_colors[:, ::-1]
        actual_colors_std = actual_colors_std[::-1]
        print('   detected card is rotated')
        card_orientation = False

    true_colors = colorbalance.ColorCheckerRGB_CameraTrax
    Check = True
    if any(actual_colors_std>40):
        print('   Some colors on the colorcard seem corrupted :(')
        card_damaged = True
    actual_colors2 = actual_colors
    iter = 0
    while Check:
        iter = iter + 1
        color_alpha, color_constant, color_gamma = colorbalance.get_color_correction_parameters(true_colors,actual_colors2,'gamma_correction')
        corrected_colors = colorbalance._gamma_correction_model(actual_colors2, color_alpha, color_constant, color_gamma)
        diff_colors = true_colors - corrected_colors
        errors = np.sqrt(np.sum(diff_colors * diff_colors, axis=0)).tolist()
        if Acc > 0.4 and np.mean(errors) > 40 and iter < 3:
            actual_colors2 = actual_colors + np.random.rand(3,24)
            print('   Corrction error high, checking again....!')
        else:
            Check = False
   
   
    if np.mean(errors) > 50:  # equivalent to 20% error
        print('   Image correction unsatisfactory!')
    
    corection_error = round((np.mean(errors)/255)*10000)/float(100)
    return(card_orientation, card_damaged, corection_error)



    