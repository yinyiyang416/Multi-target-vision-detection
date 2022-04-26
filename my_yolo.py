from cmath import atan
from msilib import add_tables
import numpy as np
import math as ma
#通过相机成像原理计算物体相对相机中心点的偏离角和直线距离
"""
x:物体在相片中的中心的x坐标相对位置
y:物体在相片中的中中心y坐标相对位置
w:物体的相对宽度
h:物体的相对高度
width:照片的真实宽度
height:照片的真实高度
f:相机的焦距
p:相机的俯仰角
H:相机的高度
"""
def angledetect(x,y,w,h,width,height,f,p,H_cam):

    #计算角度
    x_temp = x - 0.5
    angle = ma.atan(x_temp * width / f)

    #计算距离
    y_temp = y + (h / 2)#检测目标底部相对位置
    dp = y_temp * height
    p_temp = (90 - p) * ma.pi / 180
    h_temp = height / 2
    b = ma.atan((h_temp- dp)/ f)
    #print(dp,b,a)
    distance = H_cam * (ma.tan(p_temp + b))
    #print(ma.tan())
    return angle,distance

#通过相机成像原理计算物体相对相机中心点的相对位置
#参数同上
def obj_detect(x,y,w,h,width,height,f,p,H_cam):
    #计算角度
    x_temp = x + (w / 2) - 0.5
    angle = ma.atan(x_temp * width / f)
    #计算距离
    y_temp = y + (h / 2)#检测目标底部相对位置
    dp = y_temp * height
    p_temp = (90 - p) * ma.pi / 180
    h_temp = height / 2
    b = ma.atan((h_temp- dp)/ f)
    #print(dp,b,a)
    distance = H_cam * (ma.tan(p_temp + b))
    #计算相对位置
    outputx = distance * ma.tan(angle)
    outputy = distance
    return outputx,outputy

def obj_detect_line(x,y,w,h,width,height,f,p,H_cam):
    #计算角度
    x_temp1 = x - 0.5
    angle1 = ma.atan(x_temp1 * width / f)
    x_temp2 = x + w - 0.5
    angle2 = ma.atan(x_temp2 * width / f)
    #计算距离
    y_temp = y + (h / 2)#检测目标底部相对位置
    dp = y_temp * height
    p_temp = (90 - p) * ma.pi / 180
    h_temp = height / 2
    b = ma.atan((h_temp- dp)/ f)
    distance = H_cam * (ma.tan(p_temp + b))
    #计算相对位置
    outputx1 = distance * ma.tan(angle1)
    outputx2 = distance * ma.tan(angle2)
    outputy = distance
    return outputx1,outputy,outputx2,outputy,
# # #文件路径，文件格式是yolo保存文件的格式label,x,y,w,h
# txt_name = "./exp16/labels/img0.txt"
# file = open(txt_name)
# line = file.readline().strip()
# #传感器的大小，单位米
# width = 0.075
# height = 0.050
# #焦距，单位米
# f = 0.05
# #俯仰角，单位度
# p = 0
# #相机离地高度，单位米
# H_cam = 1
# while line:
#     word = line.split()
#     Angle,Distance = obj_detect(float(word[1]),float(word[2]),float(word[3]),float(word[4]),width,height,f,p,H_cam)
#     print(word[0],Angle,Distance)
#     line = file.readline().strip()
# ma.tan
# file.close()



