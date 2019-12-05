import cv2
import numpy as np
import itertools
import math
from mergeLines import merge_lines
import sudoku

NUM_LINES = 70
class Intersection:
    def __init__(self, x, y, lines):
        self.point = (x, y)
        self.lines = lines
    def __repr__(self):
        return f"({self.point[0]},{self.point[1]})"
class Line():
    def __init__(self, p1, p2):
        if p1[0] < p2[0]:
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
        else:
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
        try:
            self.slope = dy/dx
        except ZeroDivisionError:
            self.slope = dy/(dx + .0001)
        
        self.b = p1[1] - (self.slope * p1[0])
        #print(f"x1={p1[0]} y1={p1[1]} x2={p2[0]} y2={p2[1]} m={self.slope} b={self.b}")

    def intersect(self, line2):
        try:
            x = (self.b - line2.b) / (line2.slope - self.slope)
            y = (self.b / self.slope - line2.b / line2.slope) / (1 / self.slope - 1 / line2.slope)
            return Intersection(int(x), int(y), (self, line2))
        except ZeroDivisionError:
            return Intersection(-10000, -10000, (None, None))
    
    def __getitem__(self, key):
        if key == 0:
            return self.slope
        elif key == 1:
            return self.b
        else:
            raise IndexError
    
    def __repr__(self):
        return f"m={self.slope} b={self.b}"

def abs_cross_product(v1, v2):
    return abs((v1[0] * v2[1]) -(v1[1] * v2[0]))


orgimg = cv2.imread("upside_down.jpg")
gray = cv2.cvtColor(orgimg,cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(orgimg, (11,11), 5)
edge_img = cv2.Canny(img, 100, 50)
corners = [[],[],[],[]]
transformpts = np.float32([[0, 0], [2000, 0], [2000, 2000], [0, 2000]])

kernel = np.ones((5,5), np.uint8)
edge_img = cv2.dilate(edge_img, kernel, iterations=1) 

#cv2.imwrite('edges.jpg',edge_img)

lines = []
raw_lines = cv2.HoughLines(edge_img,1,np.pi/180, 200)
for i in range(NUM_LINES):
    for rho,theta in raw_lines[i]:
        if rho < 0:
            raw_lines[i,0,0] = -rho
            raw_lines[i,0,1] = theta - np.pi
raw_lines = merge_lines(raw_lines, NUM_LINES)
#print(raw_lines)
for rho,theta in raw_lines:
    if rho < 0:
        raw_lines[i,0,0] = -rho
        raw_lines[i,0,1] = theta - np.pi
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 5000*(-b))
    y1 = int(y0 + 5000*(a))
    x2 = int(x0 - 5000*(-b))
    y2 = int(y0 - 5000*(a))
    lines.append(Line((x1,y1),(x2, y2)))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)



intersections = []
for combo in itertools.combinations(lines, 2):
    intersec = combo[0].intersect(combo[1])
    print(intersec)
    if np.linalg.norm(intersec.point) > 5000 or intersec.point[0] < 0 or intersec.point[1] < 0:
        continue
    intersections.append(intersec)
    # print(intersec)
    cv2.circle(img, intersec.point, 10, (255,0,0), thickness=3)

#Find two of the corners
max_area = 0
top_left = None
top_right = None
for combo in itertools.combinations(intersections, 2):
    dx = abs(combo[0].point[0] - combo[1].point[0])
    dy = abs(combo[0].point[1] - combo[1].point[1])
    area = dx * dy
    if area > max_area:
        max_area = area
        top_left = combo[0]
        top_right = combo[1]
#print(f"max area={max_area}")
cv2.circle(img, top_right.point, 15, (0, 255, 0), thickness=3)
cv2.circle(img, top_left.point, 15, (0, 255, 0), thickness=3)
corners[0] = top_left.point
corners[2] = top_right.point

last_points = []
for intersec in intersections:
    if (intersec.lines[0] in top_left.lines and intersec.lines[1] in top_right.lines) or (intersec.lines[1] in top_left.lines and intersec.lines[0] in top_right.lines):
        last_points.append(intersec)

cv2.circle(img, last_points[0].point, 10, (0, 255, 255), thickness=3)
cv2.circle(img, last_points[1].point, 10, (0, 0, 255), thickness=3)

corners[3] = last_points[0].point
corners[1] = last_points[1].point

corners = np.float32([list(elem) for elem in corners])
print(corners)
M = cv2.getPerspectiveTransform(corners, transformpts)
warp_img = cv2.warpPerspective(orgimg,M,(2000, 2000))

cv2.imwrite('houghlines5.jpg',img)
cv2.imwrite('warp.jpg', warp_img)