import cv2
import numpy as np
import itertools
import math
from mergeLines import merge_lines

NUM_LINES = 30

class Line():
    def __init__(self, p1, p2):
        if p1[0] < p2[0]:
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
        else:
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
        self.slope = dy/dx
        self.b = p1[1] - (self.slope * p1[0])
        #print(f"x1={p1[0]} y1={p1[1]} x2={p2[0]} y2={p2[1]} m={self.slope} b={self.b}")

    def intersect(self, line2):
        try:
            x = (self.b - line2.b) / (line2.slope - self.slope)
            y = (self.b / self.slope - line2.b / line2.slope) / (1 / self.slope - 1 / line2.slope)
            return (int(x), int(y))
        except ZeroDivisionError:
            return (-10000, -10000)
    
    def __getitem__(self, key):
        if key == 0:
            return self.slope
        elif key == 1:
            return self.b
        else:
            raise IndexError
    
    def __repr__(self):
        return f"m={self.slope} b={self.b}"


img = cv2.imread("sudoku-paper-puzzle.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (11,11), 5)
edge_img = cv2.Canny(img, 100, 50)

kernel = np.ones((5,5), np.uint8)
edge_img = cv2.dilate(edge_img, kernel, iterations=1) 
cv2.imwrite('edges.jpg',edge_img)

lines = []
raw_lines = cv2.HoughLines(edge_img,1,np.pi/180, 200)
for i in range(NUM_LINES):
    for rho,theta in raw_lines[i]:
        if rho < 0:
            raw_lines[i,0,0] = -rho
            raw_lines[i,0,1] = theta - np.pi
raw_lines = merge_lines(raw_lines, NUM_LINES)
print(raw_lines)
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
    if np.linalg.norm(intersec) > 5000:
        continue
    intersections.append(intersec)
    # print(intersec)
    cv2.circle(img, intersec, 10, (255,0,0), thickness=3)

max_area = 0
top_left = None
top_right = None
for combo in itertools.combinations(intersections, 2):
    dx = abs(combo[0][0] - combo[1][0])
    dy = abs(combo[0][1] - combo[1][1])
    area = dx * dy
    if area > max_area:
        max_area = area
    top_left = combo[0]
    top_right = combo[1]
cv2.circle(img, (1000, 500), 10, (0, 255, 0), thickness=3)
cv2.circle(img, top_left, 10, (0, 255, 0), thickness=3)
cv2.imwrite('houghlines5.jpg',img)