import cv2 as cv
from cv2 import KeyPoint_convert
from asyncio.windows_events import NULL



# 绘制匹配结果
def draw_match(img1, img2, kp1, kp2, match):
    outimage = cv.drawMatches(img1, kp1, img2, kp2, match, outImg=None)
    cv.imshow("Match Result", outimage)
    cv.waitKey(0)

def ORB_match(img1, img2, inputkp1=NULL, inputkp2=NULL):
    # 初始化ORB
    orb = cv.ORB_create()
    # 有输入特征点直接使用，否则提取特征点
    if inputkp1 == NULL:
        kp1 = orb.detect(img1, None)
    else:
        kp1 = inputkp1
    if inputkp2 == NULL:
        kp2 = orb.detect(img2, None)
    else:
        kp2 = inputkp2
    # 计算描述符
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)
    # 初始化 BFMatcher
    bf = cv.BFMatcher(cv.NORM_HAMMING)
    # 对描述子进行匹配
    matches = bf.match(des1, des2)
    # 计算最大距离和最小距离
    min_distance = matches[0].distance
    max_distance = matches[0].distance
    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance
    # 筛选匹配点
    good_match = []
    for x in matches:
        if x.distance <= max(1.5 * min_distance, 30):
            good_match.append(x)
    # 绘制匹配结果
    draw_match(img1, img2, kp1, kp2, good_match)
    return kp1, kp2, good_match

# 使用自己创建的特征点集SURF进行特征点匹配
def SIFT_match(img1, img2, inputkp1=NULL, inputkp2=NULL):
    sift = cv.SIFT_create()
    # 有输入特征点直接使用，否则提取特征点
    if inputkp1 == NULL:
        kp1, des1 = sift.detectAndCompute(img1, None)
    else:
        print('get kp1')
        kp1 = inputkp1
    if inputkp2 == NULL:
        kp2 = sift.detect(img2, None)
    else:
        print('get kp2')
        kp2 = inputkp2
    # 计算描述值
    kp1, des1 = sift.compute(img1, kp1)
    kp2, des2 = sift.compute(img2, kp2)

    # 特征点匹配
    bf = cv.BFMatcher(cv.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    # 提取好匹配
    good_match = []

    for m, n in matches:
        if m.distance < 0.5*n.distance:
            good_match.append(m)

    # 绘制匹配结果

    draw_match(img1, img2, kp1, kp2, good_match)
    return kp1, kp2, good_match

# img1 = cv.imread('./env2/0000000001.png')
# img2 = cv.imread('./env2/0000000021.png')
# SIFT_match(img1,img2)