from asyncio.windows_events import NULL
from cProfile import label
from cgi import test
from typing import DefaultDict
import babel

import cv2 as cv
from cv2 import KeyPoint_convert
import numpy as np
import math as ma
#from pyrsistent import T
from scipy.spatial.transform import Rotation
from my_yolo import obj_detect

# 绘制匹配结果


def draw_match(img1, img2, kp1, kp2, match):
    outimage = cv.drawMatches(img1, kp1, img2, kp2, match, outImg=None)
    cv.imshow("Match Result", outimage)
    cv.waitKey(0)

# 从文档中提取特征点坐标和世界坐标


def read_file(file_path):
    f = open(file_path, 'r')
    data_lists = f.readlines()
    dataset = []
    datatemp = []  # 前三个为世界坐标，后三个为像素坐标
    for data in data_lists:
        data1 = data.strip('\n')
        data2 = data1.split(' ')
        datatemp = []
        for data3 in data2:
            datatemp.append(float(data3.strip('(,)')))  # 去除掉符号
        dataset.append(datatemp)
    return dataset

# 坐标转换keppoint类型


def label_to_kp(labels):
    keppoints = []
    for label in labels:
        keppoints.append(cv.KeyPoint(label[3], label[4], 1))
    return keppoints

# 拆分坐标


def label_change(labels):
    img_label = []
    world_label = []
    for label in labels:
        world_label.append(label[0:3])
        img_label.append(label[3:5])
    return img_label, world_label


def ORB_Feature(img1, img2, inputkp1=NULL, inputkp2=NULL):

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
    # 画出关键点
    # outimg1 = cv.drawKeypoints(img1, keypoints=kp1, outImage=None)
    # outimg2 = cv.drawKeypoints(img2, keypoints=kp2, outImage=None)
    # # 显示关键点
    # outimg3 = np.hstack([outimg1, outimg2])
    # cv.imshow("ORB Key Points", outimg3)
    # cv.waitKey(0)
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
        if x.distance <= max(2 * min_distance, 30):
            good_match.append(x)

    # 绘制匹配结果
    draw_match(img1, img2, kp1, kp2, good_match)
    return kp1, kp2, good_match


# 使用自己创建的特征点集SURF进行特征点匹配
def SURF_match(img1, img2, inputkp1=NULL, inputkp2=NULL):
    sift = cv.SIFT_create()
    # cv.imshow("env kp", img1)
    # cv.waitKey(0)
    # cv.imshow("env kp", img2)
    # cv.waitKey(0)
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

# 环境特征点坐标，深度图像，相机世界坐标，返回特征点世界坐标
# env_label是像素位置，deep_img是深度图片,cam_pose环境相机外参(前三个是位置，后三个欧拉角),K是内参


def world_label(env_label, deep_img, cam_pose, K):
    env_label = [int(env_label[0]), int(env_label[1])]
    deep = deep_img[env_label[1], env_label[0]]
    d = deep[0]
    d = d / 1.25  # 这个是在深度相机距离为0-200时的变化值
    cam_p = [(env_label[0] - K[0][2])/K[0][0],
             (env_label[1] - K[1][2])/K[1][1]]
    temp_p = [d * cam_p[0], d * cam_p[1], d]
    p = cam_pose[4]
    p = p * np.pi / 180
    temp_p2 = [temp_p[0]*ma.cos(p) + temp_p[2]*ma.sin(p),
               temp_p[2]*ma.cos(p) - temp_p[0]*ma.sin(p)]
    wld_p = [cam_pose[0] + temp_p2[0], cam_pose[1] +
             temp_p[1], cam_pose[2]+temp_p2[1]]
    return wld_p

# 读取环境信息，输入文件夹名字，返回图像深度图像，相机位姿


def read_env(filename):
    env_img = []
    deep_img = []
    cam_pose = read_file(filename + '/cam_pose.txt')
    for i in range(8):
        env_img.append(cv.imread(filename + '/env_img' + str(i) + '.png'))
        deep_img.append(cv.imread(filename + '/env_deep' + str(i) + '.png'))
    return env_img, deep_img, cam_pose

# 通过环境图像和输入图像计算PNP，返回被测相机的位姿欧拉角和位移向量
# img：被测相机图像
# env_img,deep_env_img：环境图像和深度图像
# env_label:环境相机位姿（位姿+欧拉角）
# cam_mat，dist_co:相机内参


def PnP_deep(img, env_img, deep_env_img, env_label, camera_matrix, dist_coeffs):

    # 使用ORB方法匹配
    #img_kp,env_kp,matchs = ORB_Feature(img,env_img,NULL,NULL)

    # 使用SURF方法匹配
    img_kp, env_kp, matchs = SURF_match(img, env_img, NULL, NULL)

    # 显示深度图
    # show_img = np.hstack([deep_env_img, env_img])
    # cv.imshow("ORB Key Points", show_img)
    # cv.waitKey(0)

    # 得到3d-2d点对
    image_points = []
    world_points = []
    for match in matchs:
        temp_label = [int(env_kp[match.trainIdx].pt[0]),
                      int(env_kp[match.trainIdx].pt[1])]
        temp_deep = deep_env_img[temp_label[1], temp_label[0]]
        # 去除深度相机没有检测到点
        if(temp_deep[0] >= 255):
            continue
        image_points.append(img_kp[match.queryIdx].pt)
        world_points.append(world_label(
            env_kp[match.trainIdx].pt, deep_env_img, env_label, camera_matrix))
    # 匹配对数量
    # print('match:')
    # print(len(world_points))
    # 使用普通PnP
    # su, R, t = cv.solvePnP(np.array(world_points),np.array(image_points),\
    # camera_matrix,dist_coeffs,flags=cv.SOLVEPNP_ITERATIVE)
    ret, rvec12, tvec12, inliers = cv.solvePnPRansac(np.array(world_points), np.array(image_points),
                                                     camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_EPNP, reprojectionError=3, iterationsCount=1000)

    # 把摄像机坐标系到世界坐标系的欧拉角和位移矩阵转化为世界坐标系到摄像机坐标系的
    rvec21, tvec21 = RT_change(np.mat(rvec12), np.mat(tvec12))
    # 弧度转化角度
    for i in range(len(rvec21)):
        rvec21[i] = rvec21[i] / ma.pi * 180
    # 输出
    if(inliers is None):
        print('error')
    else:
        print('rvec(change to degree):')
        print(rvec21.T)
        print('tvec:')
        print(tvec21.T)
    return rvec21, tvec21

# 调用PnP_deep,计算PNP
# img：被测相机图像
# env_imgs,deep_env_imgs：环境图像和深度图像列表
# env_labels:环境相机位姿列表（位姿+欧拉角）
# camera_matrix,dist_coeffs:相机内参


def PnP(img, env_imgs, deep_env_imgs, env_labels, camera_matrix, dist_coeffs):
    # 在所用环境图片中找到合适的一个
    img_index = find_goodimg(img, env_imgs)
    rvec, tvec = PnP_deep(img, env_imgs[img_index], deep_env_imgs[img_index],
                          env_labels[img_index], camera_matrix, dist_coeffs)
    return rvec, tvec

# 找到最好的环境图片


def find_goodimg(img, env_imgs):
    # 初始化，限制最大特征点
    sift = cv.SIFT_create(200)
    bf = cv.BFMatcher(cv.NORM_L2)
    match_score = []
    kp1, des1 = sift.detectAndCompute(img, None)
    for env_img in env_imgs:
        kp2, des2 = sift.detectAndCompute(env_img, None)
        matches = bf.knnMatch(des1, des2, k=2)
        good_match = []
        for m, n in matches:
            if m.distance < 0.5*n.distance:
                good_match.append(m)
        match_score.append(len(good_match))
    return match_score.index(max(match_score))

# 把位姿1到位姿2的欧拉角和位移矩阵转化为2到1的
# 输入为两个3x1的numpy矩阵


def RT_change(rvec12, tvec12):
    # 把欧拉角转化为旋转矩阵
    tempRot = Rotation.from_euler('xyz', rvec12.T)
    R12 = np.mat(tempRot.as_matrix())
    R21 = R12.T
    tvec21 = -(R21 * tvec12)
    rvec21 = -(rvec12)
    return rvec21, tvec21

# 结果可视化


def draw_pic(rveclist, tveclist,wldlist):
    import matplotlib.pyplot as plt
    #画出汽车位置
    plt.figure()
    ax = plt.axes()
    print(len(rveclist))
    for i in range(len(rveclist)):
        rvec = rveclist[i]
        tvec = tveclist[i]
        ax.arrow(tvec[0,0], tvec[2,0], 1*ma.sin(rvec[1,0]/180*ma.pi),
                 2*ma.cos(rvec[1,0]/180*ma.pi), length_includes_head=False, head_width=1, fc='r', ec='k')
        ax.plot(tvec[0,0],tvec[2,0],'ro')
        # ax.arrow(float(tveclist[i][0]), float(tveclist[i][2]), 2*ma.sin(float(rveclist[i][1])/180*np.pi),
        #          2*ma.cos(float(rveclist[i][1])/180*np.pi), length_includes_head=False, head_width=2, fc='r', ec='k')
    #画出检测位置点
    for wld_p in wldlist:
        x = wld_p[2]
        y = wld_p[4]
        label = wld_p[1]
        pic = wld_p[0]
        if(pic == 0):
            ax.plot(x,y,'ro')
        elif(pic == 1):
            ax.plot(x,y,'bo')
        elif(pic == 2):
            ax.plot(x,y,'r+')
        else:
            ax.plot(x,y,'b+')
        # if(label == 2.0):
        #     ax.plot(x,y,'ro')
        # elif(label == 7.0):
        #     ax.plot(x,y,'bo')       
    ax.set_title('car position',
                 fontsize=14, fontweight='bold')
    ax.grid()
    plt.show()

# 用于对于yolo的输出数据进行处理，返回其检测到物体的世界坐标
#world_label_list,保存检测点的世界坐标，格式[图片编号，yolo物体代号，x,y,z]
def yolo_change(file_path, num_pic, rvec_list, tvec_list, K, f):
    width = 2 * K[0][2] / (K[0][0] / f)
    height = 2 * K[1][2] / (K[1][1] / f)
    world_label_list = []

    for i in range(num_pic):
        #读取yolo网络输出文件
        label_list = read_file(file_path + '/img' + str(i) + '.txt')
        #提取出需要用的数据
        rvec = rvec_list[i]
        tvec = tvec_list[i]
        pitch = rvec[0, 0]
        yaw = rvec[1, 0]
        yaw = yaw / 180 * ma.pi
        cam_pose = [tvec[0,0],tvec[1,0],tvec[2,0]]
        # cam_h = tvec[1, 0]
        cam_h = 1.0
        for label in label_list:
            if(label[0] == 2 or label[0] == 7):
            #调用函数计算相对摄像头位置,假设地面平整
                x, y = obj_detect(label[1], label[2], label[3],
                              label[4], width, height, f, pitch, cam_h)
                #转化为世界坐标
                temp_p = [x*ma.cos(yaw) + y*ma.sin(yaw), y * ma.cos(yaw) - x*ma.sin(yaw)]
                wld_p = [i,label[0],cam_pose[0] + temp_p[0],cam_pose[1], cam_pose[2] + temp_p[1]]
                world_label_list.append(wld_p)
    return world_label_list
            



if __name__ == '__main__':
    #文件路径

    # 读取汽车图片
    num_pic = 4  # 图片数量
    img_list = []
    for i in range(num_pic):
        img_list.append(cv.imread('./car_pic1' + '/img' + str(i) + '.png'))
    # 读取环境信息
    env_img, deep_img, cam_pose = read_env('./env_pic1')
    rael_img_pose = read_file('./car_pic1' + '/cam_pose.txt')
    # 相机内参，矩阵、畸变参数
    camera_matrix = np.array(
        [[500, 0, 375],
         [0, 500, 250],
         [0, 0, 1]], dtype=np.double
    )
    dist_coeffs = np.zeros((4, 1))
    f = 0.05
    rveclist = []
    tveclist = []
    # PnP
    for img in img_list:
        r_temp, t_temp = PnP(img, env_img, deep_img,
                             cam_pose, camera_matrix, dist_coeffs)
        rveclist.append(r_temp)
        tveclist.append(t_temp)

    # for i in range(len(rveclist)):
    #     print(rveclist[i])
    #     print(tveclist[i])
    wld_list = yolo_change('./yolo_env1_car0/labels', num_pic, rveclist,
                tveclist, camera_matrix, f)
    draw_pic(rveclist,tveclist,wld_list)

# 通过事先测好环境3d-2d点队计算PnP
# def PnP(img,env_img,label):
#     env_kp = label_to_kp(label)
#     imglist,wldlist = label_change(label)
#     img_kp,envkp,match = ORB_Feature(img,env_img,NULL,env_kp)
#     world_points = []
#     image_points = []
#     for m in match:

#         image_points.append(img_kp[m.queryIdx].pt)
#         world_points.append(wldlist[m.trainIdx])

#     camera_matrix = np.array(
#         [[500, 0, 375],
#         [0, 500, 250],
#         [0, 0, 1]], dtype=np.double
#     )
#     dist_coeffs = np.zeros((4, 1))
#     (su, R, t) = cv.solvePnP(np.array(world_points),np.array(image_points),camera_matrix,dist_coeffs,flags=cv.SOLVEPNP_ITERATIVE)
#     if su:
#         print('R:')
#         print(R)
#         print('t:')
#         print(t)

# def test_pnp(label):
#     img_list,wldlist = label_change(label)
#     camera_matrix = np.array(
#         [[500, 0, 375],
#         [0, 500, 250],
#         [0, 0, 1]], dtype=np.double
#     )
#     dist_coeffs = np.zeros((4, 1))
#     ret, rvec12, tvec12,inliers = cv.solvePnPRansac(np.array(wldlist),np.array(img_list),camera_matrix,dist_coeffs,flags=cv.SOLVEPNP_EPNP,reprojectionError=3,iterationsCount=1000)
#     #把摄像机坐标系到世界坐标系的欧拉角和位移矩阵转化为世界坐标系到摄像机坐标系的
#     rvec21,tvec21 = RT_change(np.mat(rvec12),np.mat(tvec12))
#     #弧度转化角度
#     temprvec = rvec21
#     for i in range(len(rvec21)):
#         temprvec[i] = rvec21[i] / np.pi * 180
#     if(inliers is None):
#         print('error')
#     else:
#         print('R:')
#         print(temprvec)
#         print('t:')
#         print(tvec21)
