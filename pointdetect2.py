import cv2

# 读取图片并灰度处理
a = []
b = []

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
        print(x, y)


imgpath = 'F:\lyh\muti_relative_pose_demo\data\\roadsimulate4\\yellow.png'
img = cv2.imread(imgpath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 创建SURF对象，对象参数float(4000)为阈值，阈值越高，识别的特征越小。
sift = cv2.xfeatures2d.SURF_create(float(4000))
# 将图片进行SURF计算，并找出角点keypoints，keypoints是检测关键点
# descriptor是描述符，这是图像一种表示方式，可以比较两个图像的关键点描述符，可作为特征匹配的一种方法。
keypoints, descriptor = sift.detectAndCompute(gray, None)

# cv2.drawKeypoints() 函数主要包含五个参数：
# image: 原始图片
# keypoints：从原图中获得的关键点，这也是画图时所用到的数据
# outputimage：输出
# color：颜色设置，通过修改（b,g,r）的值,更改画笔的颜色，b=蓝色，g=绿色，r=红色。
# flags：绘图功能的标识设置，标识如下：
# cv2.DRAW_MATCHES_FLAGS_DEFAULT 默认值
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
# cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
img = cv2.drawKeypoints(image=img, outImage=img, keypoints = keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT, color = (51, 163, 236))

# 显示图片
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)
cv2.waitKey(0)
print(a[0], b[0])
