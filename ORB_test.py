from cgi import test
from typing import DefaultDict

import cv2 as cv
import numpy as np

#import matplotlib.pyplot as plt

def ORB_Feature(img1, img2):

    # 初始化ORB
    orb = cv.ORB_create()

    # 寻找关键点
    kp1 = orb.detect(img1)
    kp2 = orb.detect(img2)

    # 计算描述符
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)
    print(len(kp1))
    print(des1.shape)
    # 画出关键点
    outimg1 = cv.drawKeypoints(img1, keypoints=kp1, outImage=None)
    outimg2 = cv.drawKeypoints(img2, keypoints=kp2, outImage=None)
	
	# 显示关键点

    outimg3 = np.hstack([outimg1, outimg2])
    cv.imshow("ORB Key Points", outimg2)
    cv.waitKey(0)

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
    '''
        当描述子之间的距离大于两倍的最小距离时，认为匹配有误。
        但有时候最小距离会非常小，所以设置一个经验值30作为下限。
    '''
    good_match = []
    for x in matches:
        if x.distance <= max(2 * min_distance, 30):
            good_match.append(x)

    # 绘制匹配结果
    draw_match(img1, img2, kp1, kp2, good_match)


def ORBfeature(img,label,response = 0):
    orb = cv.ORB_create()
    kp = orb.detect(img, None)
    good_point = []
    for point in kp:
        if point.response > response:
            good_point.append(point)
    
    outimg = cv.drawKeypoints(img, keypoints=good_point, outImage=None)
    # cv.imshow("SURF Key Points", outimg)
    # cv.waitKey(0)
    cv.imwrite('result2/' + label + '.png', outimg, [cv.IMWRITE_PNG_COMPRESSION, 0])
    cv.waitKey(0)
    f = open("result2/texture" + label + ".txt",'a')
    for point in good_point:
        f.write(str(point.pt[0]))
        f.write(" ")
        f.write(str(point.pt[1]))
        f.write("\n")
    f.close()

def SURF_match(img1,img2):
    sift = cv.SIFT_create()
    kp1 = sift.detect(img1, None)
    kp2 = sift.detect(img2, None)

    kp1, des1 = sift.compute(img1, kp1)
    kp2, des2 = sift.compute(img2, kp2)
    
    # BFmatcher with default parms
    bf = cv.BFMatcher(cv.NORM_L2)
    matches = bf.knnMatch(des1, des2, k = 2)
    
    good_match = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_match.append(m)
        # 绘制匹配结果
    draw_match(img1, img2, kp1, kp2, good_match)


 
def ORB_test(file,img,flags = 'ORB'):#file是环境图片路径文件，img是被测试图片,flags是检测模式
    if flags == 'ORB':
        dec = cv.ORB_create()
    elif flags == 'SURF':
        dec = cv.SIFT_create()
    #读取环境图片
    f = f = open(file,'r')
    env_img_path = f.readline()
    env_kp = ()
    i = 1
    while env_img_path:
        env_img = cv.imread(env_img_path)
        temp_kp = dec.detect(env_img)
        temp_kp, temp_des = dec.compute(env_img, temp_kp)
        f = open("result2/texture" + i + ".txt",'a')
        for point in temp_kp:
            f.write(str(point.pt[0]))
            f.write(" ")
            f.write(str(point.pt[1]))
            f.write("\n")
        f.close()
        env_img_path = f.readline()


def draw_match(img1, img2, kp1, kp2, match):
    outimage = cv.drawMatches(img1, kp1, img2, kp2, match, outImg=None)
    cv.imshow("Match Result", outimage)
    cv.waitKey(0)

if __name__ == '__main__':
    # 读取图片
    image0 = cv.imread('roadsimulate7/env_img_0.png')
    image1 = cv.imread('roadsimulate7/env_img_1.png')
    image2 = cv.imread('roadsimulate7/env_img_2.png')
    image3 = cv.imread('roadsimulate7/env_img_3.png')
    image4 = cv.imread('roadsimulate7/env_img_4.png')
    image5 = cv.imread('roadsimulate7/env_img_5.png')
    image6 = cv.imread('roadsimulate7/env_img_6.png')
    image7 = cv.imread('roadsimulate7/env_img_7.png')
    ORBfeature(image0,'env0')
    ORBfeature(image1,'env1')
    ORBfeature(image2,'env2')
    ORBfeature(image3,'env3')
    ORBfeature(image4,'env4')
    ORBfeature(image5,'env5')
    ORBfeature(image6,'env6')
    ORBfeature(image7,'env7')
    #ORB_Feature(imageyellow,imageblue)
    #ORB_Feature(imagered,imagegray)
    #SURF_match(imageyellow,imageblue)

    

