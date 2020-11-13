import math
import numpy as np
import cv2
from tensorflow import keras
from joblib import load


def sort_data(x_u, y_u, f_u):
    combined = zip(x_u, y_u, f_u)
    cs = sorted(combined, key=lambda x: x[2])

    return zip(*cs)


def predict_segs(segs, rec_feats, fn, loaded_model, scaler):
    temp_ball_seg = []
    temp_rec_feat = []
    temp_cent = []

    segs = scaler.transform(segs)  # Normalize inputs

    y_score = np.round(loaded_model.predict(segs), 6)
    y_score = np.reshape(y_score, len(y_score))

    y_pred = (loaded_model.predict(segs) > 0.1).astype("int32")
    y_pred = np.reshape(y_pred, len(y_pred))  # Reshape y pred


    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            temp_ball_seg.append(segs[i])
            temp_rec_feat.append(rec_feats[i])
            temp_cent.append([rec_feats[i][4], 720 - rec_feats[i][5], fn, y_score[i]])

    return temp_cent



def frame_to_segment(frame, bs_orig, xl, xr, yu, yd):
    frame_segments = []
    frame_rec_feat = []
    frame_ball_cont = []
    c_frame = frame.copy()
    con_frame = frame.copy()

    cv2.rectangle(con_frame, (xl, yd), (xr, yu), (0, 255, 0), 2)
    g_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # Convert to grayscale
    #
    # gb_frame = cv2.GaussianBlur(g_frame, (5, 5), 0)     # Gaussian Blur to reduce noise
    #
    # bs_img = bs_obj.apply(gb_frame)                     # applying background subtraction on each frame resulting in a foreground mask

    bs_img = bs_orig
    # Morphological Opening
    kernel = np.ones((5, 5), np.uint8)                  # set kernel as 5x5 matrix from numpy
    dilation_image = cv2.dilate(bs_img, kernel, iterations=1)
    erosion_image = cv2.erode(dilation_image, kernel, iterations=1)



    # ----------------------------------------------------------------------------------#
    # Edge Detection
    laplacian = cv2.Laplacian(src=erosion_image, ddepth=cv2.CV_8U, ksize=5)


    # ----------------------------------------------------------------------------------#
    # Contours
    contours, hierarchy = cv2.findContours(laplacian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(con_frame, contours, -1, (0, 255, 0), 3)
    for contour in contours:
        cnt = contour

        # Calculate filled area by filling each contour and counting non-zero pixels
        mask = np.zeros(g_frame.shape, np.uint8)
        cv2.fillConvexPoly(mask, cnt, 255, lineType=8, shift=0)
        area = cv2.countNonZero(mask)

        # Find outer area (perimeter area) using the pre-existing function
        outer_area = cv2.contourArea(contour)

        # Find the Rotated Bounded Rectangle that would fit this contour
        minrec = cv2.minAreaRect(contour)
        x_c = minrec[0][0]
        y_c = minrec[0][1]



        in_crop = (y_c < yu) & (y_c > yd) & (x_c < xr) & (x_c > xl)
        if (area > 5) & (area < 600) & in_crop:
            size = 16
            hs = size / 2
            x0 = math.floor(minrec[0][0] - hs)
            y0 = math.floor(minrec[0][1] + hs)
            x1 = math.floor(minrec[0][0] + hs)
            y1 = math.floor(minrec[0][1] - hs)

            segment = np.ones([size, size], dtype=np.uint8)

            for i in range(0, size):
                for j in range(0, size):

                    # inframe
                    if (y0 - i <= 719) & (y0 - i >= 0) & (x0 + j <= 1279) & (x0 + j >= 0):
                        segment[15 - i][j] = g_frame[y0 - i][x0 + j]

                    # outframe - top & bottom
                    elif (y0 - i > 719) & (x0 + j <= 1279) & (x0 + j >= 0):
                        segment[15 - i][j] = g_frame[719][x0 + j]
                    elif (y0 - i < 0) & (x0 + j <= 1279) & (x0 + j >= 0):
                        segment[15 - i][j] = g_frame[0][x0 + j]

                    # outframe - right & left
                    elif (x0 + j > 1279) & (y0 - i <= 719) & (y0 - i >= 0):
                        segment[15 - i][j] = g_frame[y0 - i][1279]

                    elif (x0 + j < 0) & (y0 - i <= 719) & (y0 - i >= 0):
                        segment[15 - i][j] = g_frame[y0 - i][0]

            #cv2.rectangle(rframe, (rec[0], rec[1]), (rec[2], rec[3]), (0, 255, 0), 2)
            frame_segments.append(np.ndarray.flatten(segment))
            frame_rec_feat.append([x0, y0, x1, y1, minrec[0][0], minrec[0][1]])
            frame_ball_cont.append(contour)

    # cv2.imshow('bs', bs_img)
    # cv2.imshow('ero', erosion_image)
    # cv2.imshow('lapy', laplacian)
    # cv2.imshow('o', c_frame)
    # cv2.imshow('con', con_frame)
    # key1 = cv2.waitKey(0)
    return frame_segments, frame_rec_feat

def plot_traj(trajectory):
    xp = []
    yp = []
    fp = []

    for c, cen in enumerate(trajectory):
        for d in range(len(cen)):
            xp.append(cen[d][0])
            yp.append(720 - cen[d][1])
            fp.append(cen[d][2])

    return (xp, yp, fp)


def line_build(p1, p2):
    M, C = 0, 0
    Vert, Hor, Normal = False, False, False

    # create a line between using two points given
    run = p2[0] - p1[0]
    rise = p2[1] - p1[1]

    if (run == 0) and (run == 0):  # duplicate point (no line)
        pass
    elif run == 0:  # vertical line
        Vert = True
    elif rise == 0:  # horizontal line
        Hor = True

    else:  # normal line
        M = rise / run  # M = (y1-y0)/(x1-x0)
        C = p1[1] - M * p1[0]  # C = y-Mx
        Normal = True

    return M, C, Normal, Vert, Hor


def match_line(cen, p1, M, C, Normal, Vert, Hor, tl, tc):
    match = False
    # if centres[l][0][0] > 0:
    #     fn = centres[l][0][2]
    #     for cen in centres[l]:
    dline = tl+1

    if Normal:
        yline = M * (cen[0]) + C
        xline = (cen[1] - C) / M

        xl_x = abs(xline - cen[0])
        yl_y = abs(yline - cen[1])

        if xl_x == 0:  # if either xl_x or yl_y is 0 then the other one will be too and it means the point is exactly on the line
            dline = 0

        else:
            theta = math.atan(yl_y / xl_x)
            dline = yl_y * math.cos(theta)

    elif Hor:
        yline = p1[1]
        dline = abs(yline - cen[1])
    elif Vert:
        xline = p1[0]
        dline = abs(xline - cen[0])

    dx = np.abs(cen[0] - p1[0])
    dy = np.abs(cen[1] - p1[1])
    dr = np.sqrt(dx ** 2 + dy ** 2)

    if (dline < tl) and (dr < tc ):
        match = True
        dscore = dline+dr
        return cen, match, dscore

    else:
        return cen, match, 1000
