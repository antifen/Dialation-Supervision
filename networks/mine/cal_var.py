import numpy as np


# binary image thinning (skeletonization) in-place.
# implements Zhang-Suen algorithm.
# http://agcggs680.pbworks.com/f/Zhan-Suen_algorithm.pdf
# @param im   the binary image
def thinningZS(im):
    prev = np.zeros(im.shape, np.uint8);
    while True:
        im = thinningZSIteration(im, 0);
        im = thinningZSIteration(im, 1)
        diff = np.sum(np.abs(prev - im));
        if not diff:
            break
        prev = im
    return im

# pass of Zhang-Suen thinning
def thinningZSIteration(im, iter):
    marker = np.zeros(im.shape, np.uint8);
    for i in range(1, im.shape[0] - 1):
        for j in range(1, im.shape[1] - 1):
            p2 = im[(i - 1), j];
            p3 = im[(i - 1), j + 1];
            p4 = im[(i), j + 1];
            p5 = im[(i + 1), j + 1];
            p6 = im[(i + 1), j];
            p7 = im[(i + 1), j - 1];
            p8 = im[(i), j - 1];
            p9 = im[(i - 1), j - 1];
            A = (p2 == 0 and p3) + (p3 == 0 and p4) + \
                (p4 == 0 and p5) + (p5 == 0 and p6) + \
                (p6 == 0 and p7) + (p7 == 0 and p8) + \
                (p8 == 0 and p9) + (p9 == 0 and p2);
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            m1 = (p2 * p4 * p6) if (iter == 0) else (p2 * p4 * p8);
            m2 = (p4 * p6 * p8) if (iter == 0) else (p2 * p6 * p8);

        if (A == 1 and (B >= 2 and B <= 6) and m1 == 0 and m2 == 0):
            marker[i, j] = 1;

    return np.bitwise_and(im, np.bitwise_not(marker))

def thinningSkimage(im):
    from skimage.morphology import skeletonize
    return skeletonize(im).astype(np.uint8)

def thinning(im):
    try:
        return thinningSkimage(im)
    except:
        return thinningZS(im)

# check if a region has any white pixel
def notEmpty(im, x, y, w, h):
    return np.sum(im) > 0

# merge ith fragment of second chunk to first chunk
# @param c0   fragments from first  chunk
# @param c1   fragments from second chunk
# @param i    index of the fragment in first chunk
# @param sx   (x or y) coordinate of the seam
# @param isv  is vertical, not horizontal?
# @param mode 2-bit flag,
#             MSB = is matching the left (not right) end of the fragment from first  chunk
#             LSB = is matching the right (not left) end of the fragment from second chunk
# @return     matching successful?
#
def mergeImpl(c0, c1, i, sx, isv, mode):
    B0 = (mode >> 1 & 1) > 0;  # match c0 left
    B1 = (mode >> 0 & 1) > 0;  # match c1 left
    mj = -1;
    md = 4;  # maximum offset to be regarded as continuous

    p1 = c1[i][0 if B1 else -1];

    if (abs(p1[isv] - sx) > 0):  # not on the seam, skip
        return False

    # find the best match
    for j in range(len(c0)):
        p0 = c0[j][0 if B0 else -1];
        if (abs(p0[isv] - sx) > 1):  # not on the seam, skip
            continue

        d = abs(p0[not isv] - p1[not isv]);
        if (d < md):
            mj = j;
            md = d;

    if (mj != -1):  # best match is good enough, merge them
        if (B0 and B1):
            c0[mj] = list(reversed(c1[i])) + c0[mj]
        elif (not B0 and B1):
            c0[mj] += c1[i]
        elif (B0 and not B1):
            c0[mj] = c1[i] + c0[mj]
        else:
            c0[mj] += list(reversed(c1[i]))

        c1.pop(i);
        return True;
    return False;
HORIZONTAL = 1;
VERTICAL = 2;

def mergeFrags(c0, c1, sx, dr):
    for i in range(len(c1) - 1, -1, -1):
        if (dr == HORIZONTAL):
            if (mergeImpl(c0, c1, i, sx, False, 1)): continue;
            if (mergeImpl(c0, c1, i, sx, False, 3)): continue;
            if (mergeImpl(c0, c1, i, sx, False, 0)): continue;
            if (mergeImpl(c0, c1, i, sx, False, 2)): continue;
        else:
            if (mergeImpl(c0, c1, i, sx, True, 1)): continue;
            if (mergeImpl(c0, c1, i, sx, True, 3)): continue;
            if (mergeImpl(c0, c1, i, sx, True, 0)): continue;
            if (mergeImpl(c0, c1, i, sx, True, 2)): continue;

    c0 += c1

def chunkToFrags(im, x, y, w, h):
    frags = []
    on = False;  # to deal with strokes thicker than 1px
    li = -1;
    lj = -1;

    # walk around the edge clockwise
    for k in range(h + h + w + w - 4):
        i = 0;
        j = 0;
        if (k < w):
            i = y + 0;
            j = x + k;
        elif (k < w + h - 1):
            i = y + k - w + 1;
            j = x + w - 1;
        elif (k < w + h + w - 2):
            i = y + h - 1;
            j = x + w - (k - w - h + 3);
        else:
            i = y + h - (k - w - h - w + 4);
            j = x + 0;

        if (im[i, j]):  # found an outgoing pixel
            if (not on):  # left side of stroke
                on = True;
                frags.append([[j, i], [x + w // 2, y + h // 2]])
        else:
            if (on):  # right side of stroke, average to get center of stroke
                frags[-1][0][0] = (frags[-1][0][0] + lj) // 2;
                frags[-1][0][1] = (frags[-1][0][1] + li) // 2;
                on = False;
        li = i;
        lj = j;

    if (len(frags) == 2):  # probably just a line, connect them
        f = [frags[0][0], frags[1][0]];
        frags.pop(0);
        frags.pop(0);
        frags.append(f);
    elif (len(frags) > 2):  # it's a crossroad, guess the intersection
        ms = 0;
        mi = -1;
        mj = -1;
        # use convolution to find brightest blob
        for i in range(y + 1, y + h - 1):
            for j in range(x + 1, x + w - 1):
                s = \
                    (im[i - 1, j - 1]) + (im[i - 1, j]) + (im[i - 1, j + 1]) + \
                    (im[i, j - 1]) + (im[i, j]) + (im[i, j + 1]) + \
                    (im[i + 1, j - 1]) + (im[i + 1, j]) + (im[i + 1, j + 1]);
                if (s > ms):
                    mi = i;
                    mj = j;
                    ms = s;
                elif (s == ms and abs(j - (x + w // 2)) + abs(i - (y + h // 2)) < abs(mj - (x + w // 2)) + abs(
                        mi - (y + h // 2))):
                    mi = i;
                    mj = j;
                    ms = s;

        if (mi != -1):
            for i in range(len(frags)):
                frags[i][1] = [mj, mi]
    return frags;

def traceSkeleton(im, x, y, w, h, csize, maxIter, rects):
    frags = []

    if (maxIter == 0):  # gameover
        return frags;
    if (w <= csize and h <= csize):  # recursive bottom
        frags += chunkToFrags(im, x, y, w, h);
        return frags;

    ms = im.shape[0] + im.shape[1];  # number of white pixels on the seam, less the better
    mi = -1;  # horizontal seam candidate
    mj = -1;  # vertical   seam candidate

    if (h > csize):  # try splitting top and bottom
        for i in range(y + 3, y + h - 3):
            if (im[i, x] or im[(i - 1), x] or im[i, x + w - 1] or im[(i - 1), x + w - 1]):
                continue

            s = 0;
            for j in range(x, x + w):
                s += im[i, j];
                s += im[(i - 1), j];

            if (s < ms):
                ms = s;
                mi = i;
            elif (s == ms and abs(i - (y + h // 2)) < abs(mi - (y + h // 2))):
                # if there is a draw (very common), we want the seam to be near the middle
                # to balance the divide and conquer tree
                ms = s;
                mi = i;

    if (w > csize):  # same as above, try splitting left and right
        for j in range(x + 3, x + w - 2):
            if (im[y, j] or im[(y + h - 1), j] or im[y, j - 1] or im[(y + h - 1), j - 1]):
                continue

            s = 0;
            for i in range(y, y + h):
                s += im[i, j];
                s += im[i, j - 1];
            if (s < ms):
                ms = s;
                mi = -1;  # horizontal seam is defeated
                mj = j;
            elif (s == ms and abs(j - (x + w // 2)) < abs(mj - (x + w // 2))):
                ms = s;
                mi = -1;
                mj = j;

    nf = [];  # new fragments
    if (h > csize and mi != -1):  # split top and bottom
        L = [x, y, w, mi - y];  # new chunk bounding boxes
        R = [x, mi, w, y + h - mi];

        if (notEmpty(im, L[0], L[1], L[2], L[3])):  # if there are no white pixels, don't waste time
            if (rects != None): rects.append(L);
            nf += traceSkeleton(im, L[0], L[1], L[2], L[3], csize, maxIter - 1, rects)  # recurse

        if (notEmpty(im, R[0], R[1], R[2], R[3])):
            if (rects != None): rects.append(R);
            mergeFrags(nf, traceSkeleton(im, R[0], R[1], R[2], R[3], csize, maxIter - 1, rects), mi, VERTICAL);

    elif (w > csize and mj != -1):  # split left and right
        L = [x, y, mj - x, h];
        R = [mj, y, x + w - mj, h];
        if (notEmpty(im, L[0], L[1], L[2], L[3])):
            if (rects != None): rects.append(L);
            nf += traceSkeleton(im, L[0], L[1], L[2], L[3], csize, maxIter - 1, rects);

        if (notEmpty(im, R[0], R[1], R[2], R[3])):
            if (rects != None): rects.append(R);
            mergeFrags(nf, traceSkeleton(im, R[0], R[1], R[2], R[3], csize, maxIter - 1, rects), mj, HORIZONTAL);

    frags += nf;
    if (mi == -1 and mj == -1):  # splitting failed! do the recursive bottom instead
        frags += chunkToFrags(im, x, y, w, h);

    return frags


import numpy as np
import os
import cv2
import random
import imageio
from skimage.morphology import binary_dilation
import torch


def compute_thickness(mask, real):
    roi = mask * real
    deno = np.sum(np.sum(mask / 255.))  # the overall pixels
    return np.sum(np.sum(roi / 255.)) / deno


def cut_vessel(im0, polys, ratio=0.9, thickness=6):
    black_mask = np.zeros((im0.shape[0], im0.shape[1], 3)).astype(np.uint8)
    points = []
    thick_list = []
    for ind, l in enumerate(polys):
        temp_map = np.zeros((im0.shape[0], im0.shape[1], 3)).astype(np.uint8)
        for i in range(0, len(l) - 1):
            cv2.line(temp_map, (l[i][0], l[i][1]), (l[i + 1][0], l[i + 1][1]), [255, 255, 255], thickness)
        thick_list.append([compute_thickness(temp_map[:, :, 0], im0), ind])
    print(thick_list)
    thick_list = np.array(thick_list)
    return thick_list[:, 0]


if __name__ == "__main__":
    R = 0.3  # noise ratio
    TAU = 11  # thickness threshold
    # im0 = imageio.imread("./datasets/DRIVE/*.gif")
    # im0 = imageio.imread("./datasets/CHASEDB/*.png")
    im0 = imageio.imread("./datasets/STARE/*.pgm")
    # im0 = np.array(im0)
    # im0 =binary_dilation(im0).astype(np.uint8)
    # print(np.max(im0))
    black_map = np.zeros((im0.shape[0], im0.shape[1], 3)).astype(np.uint8)
    im = (im0[:, :] >=1 ).astype(np.uint8)
    im = thinning(im);
    rects = []
    polys = traceSkeleton(im, 0, 0, im.shape[1], im.shape[0], 10, 999, rects)
    data = cut_vessel(im, polys, R, TAU)
    var=np.std(100*data)
    print(var)