import numpy as np
from array import *
import cv2
import math
import matplotlib.pyplot as plt
from tensorflow import keras
from joblib import load
from traj_build import build_trajectories
from traj_correction import*
from functions import*
from traj_interpolation import*
from Track import track_ball
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# create video capture object
# test_data_curated0
# Squash3_curated0
# squashclip_curated0
cap = cv2.VideoCapture(r'C:\Users\User\Desktop\Work 1.0\Project 2020\Cut Squash Games\squashcclip_curated0.avi')

# initializing subtractor
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=210, nmixtures=5, backgroundRatio=0.7, noiseSigma=6)
#fgbg_c = cv2.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=5, backgroundRatio=0.7, noiseSigma=6)
#fg_GSOC = cv2.bgsegm.createBackgroundSubtractorGSOC(nSamples=100,hitsThreshold = 2 ,noiseRemovalThresholdFacBG = 0.00000004,noiseRemovalThresholdFacFG = 0.000000004)

# Initializing counters/constants
fp_thresh = 0.35
count = -1
ball_segments = []
ball_rec_feat = []
ball_cont = []
centres = []
original = []
orig_with_c = []
bs_orig_min = []

# Load Classifier
loaded_model = keras.models.load_model("model3_Test_1255.h5")
loaded_model.summary()
scaler = load('std_scaler_08nov1255.bin')

# Class Definitions

while (1):
    if count < 250:
        ret, frame = cap.read()
        rframe = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR_EXACT)
        count +=1
        gbgframe = cv2.GaussianBlur(rframe, (5, 5), 0)  # to reduce the noise
        # gbc_frame = cv2.GaussianBlur(rframe, (5, 5), 0)

        img = fgbg.apply(gbgframe)  # applying background subtraction on each frame resulting in a foreground mask
        #img_c = fgbg_c.apply(gbgframe)

    else:
        frame_rec_feat = []
        frame_ball_cont = []
        frame_segments = []


        ret, frame = cap.read()  # capturing next frame
        if not ret:       # EOF
            break

        else:
            count +=1  # incrementing counter

            rframe = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR_EXACT)
            original.append(rframe.copy())
            a_frame = rframe.copy()
            b_frame = rframe.copy()
            # cv2.imshow('original', frame)
            # cv2.imshow('resized', rframe)

            gframe = cv2.cvtColor(rframe, cv2.COLOR_BGR2GRAY)

            gbgframe = cv2.GaussianBlur(rframe, (5, 5), 0)  # to reduce the noise
            #gbc_frame = cv2.GaussianBlur(rframe, (5, 5), 0)

            img = fgbg.apply(gbgframe) # applying background subtraction on each frame resulting in a foreground mask

            #bs_orig_min.append(fgbg_min.apply(gbgframe))


            kernel = np.ones((5, 5), np.uint8)  # set kernel as 3x3 matrix from numpy

            dilation_image = cv2.dilate(img, kernel, iterations=7)
            erosion_image = cv2.erode(dilation_image, kernel, iterations=7)

            img_0 = cv2.cvtColor(erosion_image.copy(), cv2.COLOR_GRAY2BGR)
            img_1 = cv2.cvtColor(erosion_image.copy(), cv2.COLOR_GRAY2BGR)
            # cv2.imshow('bs', erosion_image)

            # ----------------------------------------------------------------------------------#
            # Edge Detection
            #laplacian = cv2.Laplacian(src=img, ddepth=cv2.CV_8U, ksize=5)

            #cv2.imshow('lapy', laplacian)

            # ----------------------------------------------------------------------------------#
            # Contours
            contours, hierarchy = cv2.findContours(erosion_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            con1, h1 = cv2.findContours(erosion_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # Copy of original upon which chosen contours will be drawn

            cv2.drawContours(img_0, contours, -1, (0,255,0), 2)
            cv2.drawContours(img_1, con1, -1, (0,255,0), 2)

            cv2.imshow('external', img_0)
            cv2.imshow('all', img_1)

            for contour in contours:
                cnt = contour

                # Calculate filled area by filling each contour and counting non-zero pixels
                mask = np.zeros(gframe.shape, np.uint8)
                cv2.fillConvexPoly(mask, cnt, 255, lineType=8, shift=0)
                area = cv2.countNonZero(mask)

                # Find outer area (perimeter area) using the pre-existing function
                outer_area = cv2.contourArea(contour)

                # Find the Rotated Bounded Rectangle that would fit this contour
                minrec = cv2.minAreaRect(contour)
                #print(minrec)
                x0 = int(minrec[0][0] - math.floor(minrec[1][0]/2))
                y0 = int(minrec[0][1] + math.floor(minrec[1][1]/2))
                x1 = int(minrec[0][0] + math.floor(minrec[1][0]/2))
                y1 = int(minrec[0][1] - math.floor(minrec[1][1] / 2))
                #print(minrec[2])

                #box = cv2.boxPoints(minrec)  # cv2.boxPoints(rect) for OpenCV 3.x
                #box = np.int0(box)
                #cv2.drawContours(aframe, [cnt], -1, (0, 255, 0), 3)
                #cv2.drawContours(aframe, [box], 0, (0, 0, 255), 2)
                #cv2.rectangle(aframe, (x0, y0), (x1, y1), (0, 0, 255), 2)

                #cv2.imshow('minrec', aframe)


                # mincir = cv2.minEnclosingCircle(contour)
                #
                # print(mincir)
                p = cv2.arcLength(cnt, True)
                p1 = len(contour)

                #print(p, p1)

                if (area > 10) & (area < 400):
                    size = 16
                    hs = size/2
                    x0 = math.floor(minrec[0][0] - hs)
                    y0 = math.floor(minrec[0][1] + hs)
                    x1 = math.floor(minrec[0][0] + hs)
                    y1 = math.floor(minrec[0][1] - hs)
                    cv2.rectangle(rframe, (x0, y0), (x1, y1), (0, 0, 255), 2)

                    segment = np.ones([size, size], dtype=np.uint8)

                    for i in range(0, size):
                        for j in range(0, size):

                            # inframe
                            if (y0 - i <= 719) & (y0 - i >= 0) & (x0 + j <= 1279) & (x0 + j >= 0):
                                segment[15 - i][j] = gframe[y0 - i][x0 + j]

                            # outframe - top & bottom
                            elif (y0 - i > 719) & (x0 + j <= 1279) & (x0 + j >= 0):
                                segment[15 - i][j] = gframe[719][x0 + j]
                            elif (y0 - i < 0) & (x0 + j <= 1279) & (x0 + j >= 0):
                                segment[15 - i][j] = gframe[0][x0 + j]

                            # outframe - right & left
                            elif (x0 + j > 1279) & (y0 - i <= 719) & (y0 - i >= 0):
                                segment[15 - i][j] = gframe[y0 - i][1279]

                            elif (x0 + j < 0) & (y0 - i <= 719) & (y0 - i >= 0):
                                segment[15 - i][j] = gframe[y0 - i][0]


                    frame_segments.append(np.ndarray.flatten(segment))
                    frame_rec_feat.append([x0, y0, x1, y1, minrec[0][0], minrec[0][1]])
                    frame_ball_cont.append(contour)

            temp_ball_segment = []
            temp_rec_feat = []
            temp_ball_cont = []
            temp_cent = []

            if len(frame_segments) > 0:
                frame_segments = scaler.transform(frame_segments)

                y_score = np.round(loaded_model.predict(frame_segments), 6)
                y_score = np.reshape(y_score, len(y_score))

                y_pred = (loaded_model.predict(frame_segments) > fp_thresh).astype("int32")
                y_pred = np.reshape(y_pred, len(y_pred))  # Reshape y pred
                scores = []
                for i in range(len(y_pred)):
                    if y_pred[i] == 1:
                        temp_ball_segment.append(frame_segments[i])
                        temp_rec_feat.append(frame_rec_feat[i])
                        scores.append(y_score[i])
                        temp_ball_cont.append(frame_ball_cont[i])

                        ball_segments.append(frame_segments[i])
                        ball_rec_feat.append(frame_rec_feat[i])
                        ball_cont.append(frame_ball_cont[i])

                if len(temp_rec_feat) > 0:
                    for r, rec in enumerate(temp_rec_feat):
                        print(rec[0], rec[1])
                        print(rec[2], rec[3])
                        cv2.rectangle(rframe, (rec[0], rec[1]), (rec[2], rec[3]), (0, 255, 0), 2)
                        temp_cent.append([rec[4], 720-rec[5], count, scores[r]])

                    centres.append(temp_cent)



                # for cont in frame_ball_cont:
                #     tempf_con = cont
                #     cv2.drawContours(aframe, [tempf_con], -1, (0, 0, 255, 1))
                # for con in temp_ball_cont:
                #     temp_con = con
                #     cv2.drawContours(aframe, [temp_con], -1, (0, 255, 0), 2)

            #cv2.imshow('orig', a_frame)
            #cv2.imshow('detections', rframe)
            #cv2.imshow('a', aframe)
            # print(count)
            print(temp_cent)
            keyVal = cv2.waitKey(0) & 0xFF

            # if (keyVal == ord('p')):
            #     keyVal1 = cv2.waitKey(0) & 0xFF
            orig_with_c.append(rframe.copy())
            if count == 1000:
                break

print('detect complete')
# np.save('originals_nodila', original)
# np.save('bs_orig_nodila', bs_orig_min)
# np.save('centres_nodila', centres)
cap.release()
cv2.destroyAllWindows()


xmax = 1280
ymax = 720
x, y, f = plot_traj(centres)

f3, (ax4, ax5) = plt.subplots(1, 2)
ax4.scatter(f, y)
ax4.set_title('Y vs Frame')
ax4.set_ylim(ymax, 0)
ax4.set_xlabel('F')
ax4.set_ylabel('Y')
ax5.scatter(f, x)
ax5.set_title('X vs Frame')
ax5.set_ylim(0, xmax)
ax5.set_xlabel('F')
ax5.set_ylabel('X')
plt.show()

# xi_s, yi_s, fi_s = track_ball(centres.copy(), original.copy(), bs_orig=bs_orig_min.copy(), model_obj=loaded_model, scaler_obj=scaler)
# print('track and plot complete')
#
# cv2.waitKey(0)
#
# count = 0
# start = True
# for fn, frm in enumerate(orig_with_c):
#     frame_count = 0
#     if count < len(fi_s):
#         if fn < fi_s[count]:
#             pass
#         elif fn > fi_s[count]:
#             pass
#         else:
#             while (count < len(fi_s)) and (fn == fi_s[count]):
#                 size = 16
#                 hs = size / 2
#                 x0 = math.floor(xi_s[count] - hs)
#                 y0 = math.floor(yi_s[count] + hs)
#                 x1 = math.floor(xi_s[count] + hs)
#                 y1 = math.floor(yi_s[count] - hs)
#                 cv2.rectangle(frm, (x0, y0), (x1, y1), (0, 255, 0), 2)
#                 count +=1
#                 frame_count +=1
#
#     # print(fn)
#     # print(count)
#     # print(frame_count)
#     cv2.imshow('final', frm)
#
#     if start:
#         keyVal = cv2.waitKey(0) & 0xFF
#         start = False
#     else:
#         keyVal = cv2.waitKey(100) & 0xFF
#
#         if (keyVal == ord('p')):
#             keyVal1 = cv2.waitKey(0) & 0xFF
#
#         elif (keyVal == 27):
#             break
#
#
# cv2.destroyAllWindows()
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# #
# # ax.scatter(framenumbers, xc, yc)
# # ax.set_ylim(0, 1280)
# # ax.set_zlim(720, 0)
# # ax.set_xlabel('Time')
# # ax.set_ylabel('X')
# # ax.set_zlabel('Y')
# #
# # plt.show()
