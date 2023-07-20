import numpy as np
import cv2
import sklearn.cluster
import time
from collections import defaultdict, Counter
from utils import *

CARD_MAX_AREA = 400000
CARD_MIN_AREA = 200000
SHAPE_MIN_AREA = 15000

BKG_THRESH = 30

class Card:
    def __init__(self):
        self.count = 0
        self.shape = 'unknown'
        self.shade = 'unknown'
        self.color = 'unknown'
        self.outer_contours = []
        self.inner_contours = []
        self.corner_points = []
        self.inner_corner_points = []
        self.card_area = 0
        self.center = (0, 0)
        self.top_left = (0, 0)
        self.bottom_right = (0, 0)

    def __str__(self):
        return f"Card({self.count}, {self.shape}, {self.shade}, {self.color}"
    
    def finish_card(self, image):
        self.count = len(self.inner_contours)
        label_counts, first_dominant, second_dominant = self.cluster_pixels(image)
        shade = self.get_shading(label_counts)
        if shade == 'full':
            color = bgr_to_color(first_dominant)
        else:
            color = bgr_to_color(second_dominant)

        self.color = color
        self.get_shape()
        self.calculate_center()

        return self
    
    def calculate_center(self):

        # ensure outer_contours is a NumPy array

        if len(self.outer_contours) > 0:
            x, y, w, h = cv2.boundingRect(self.outer_contours)
            self.center = (int(x + w/2), int(y + h/2))
            self.top_left = (x,y)
            self.bottom_right = (x+w, y+h)
        else:
            self.center = (0, 0)  # reset to (0, 0) if no contours

        return self

    def get_shape(self):
        first_contour = self.inner_contours[0]
        first_inner_corner_points = self.inner_corner_points[0]
        convexHull = cv2.convexHull(first_contour, returnPoints = False)
        convexityDefects = cv2.convexityDefects(first_contour, convexHull)
        max_defect_length = np.max([d[0][3] for d in convexityDefects])

        if len(first_inner_corner_points) == 4:
            shape = 'diamonds'
        elif max_defect_length > 2000:
            shape = 'squiggle'
        else:
            shape = 'oval'
        
        self.shape = shape

        return self

    def cluster_pixels(self, image):
        first_contour = self.inner_contours[0]
        x,y,w,h = cv2.boundingRect(first_contour)
        rect_image = image[y:y+h, x:x+w]
        all_colors = rect_image.reshape((rect_image.shape[0] * rect_image.shape[1], 3))
        clt = sklearn.cluster.KMeans(n_clusters=2, n_init=10)
        labels = clt.fit_predict(all_colors)
        label_counts = Counter(labels)
        first_dominant = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
        second_dominant = clt.cluster_centers_[label_counts.most_common(2)[1][0]]

        return label_counts, first_dominant, second_dominant

    def get_shading(self, label_counts):
        density_ratio = list(label_counts.values())[0]/list(label_counts.values())[1]
        shape = self.shape
        if shape == 'squiggle':  
            if density_ratio < 0.7:
                shade = 'full'
            elif density_ratio < 3.5:
                shade = 'striped'
            else:
                shade = 'empty'
        else:
            if density_ratio < 1.4:
                shade = 'full'
            elif density_ratio < 5.3:
                shade = 'striped'
            else:
                shade = 'empty'

        self.shade = shade

        return self

start_time = time.time()

# def pre_process(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 1)
#     img_w, img_h = np.shape(img)[:2]
#     bkg_level = gray[int(img_h/100)][int(img_w/2)]
#     thresh_level = bkg_level + BKG_THRESH

#     retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)
#     return thresh

# def get_contours(thresh):
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

#     card_list = []
#     # store outer contour indices for comparison
#     outer_indices = []
#     # store current inner contours
#     current_inner_contours = []
#     current_inner_corner_points = []
#     card_count = 0 ######## ADD COUNTER TO STOP EVENTUALLY #####

#     for i in range(len(contours)):
#         size = cv2.contourArea(contours[i])
#         if SHAPE_MIN_AREA < size < CARD_MAX_AREA:
#             peri = cv2.arcLength(contours[i], True)
#             approx = cv2.approxPolyDP(contours[i], 0.01 * peri, True)
#             pts = np.float32(approx)

#             # if the contour has no parent, it is an outer contour
#             if hierarchy[0][i][3] == -1 and size > CARD_MIN_AREA:
#                 if current_inner_contours:
#                     # we add the inner contours to the previous card because we have moved to a new outer contour
#                     card_list[-1].inner_contours += current_inner_contours
#                     card_list[-1].inner_corner_points += current_inner_corner_points  # Add inner contours' corner points to the last card
#                     current_inner_contours = []  # reset the inner contours list
#                     current_inner_corner_points = []

#                 # initialize card

#                 card = Card()
#                 card.card_area = size
#                 card.corner_points = pts
#                 card.outer_contours = contours[i]
#                 outer_indices.append(i)
#                 card_list.append(card)
#                 card_count += 1
#             elif hierarchy[0][i][3] in outer_indices:  # it's an inner contour if its parent is an outer contour
#                 current_inner_contours.append(contours[i])
#                 current_inner_corner_points.append(pts)

#     # adding the inner contours for the last card after the loop ends
#     if current_inner_contours:
#         card_list[-1].inner_contours += current_inner_contours
#         card_list[-1].inner_corner_points += current_inner_corner_points  # add inner contours' corner points to the last card
#     return card_list


# image = cv2.imread("images/15.jpg")

# height, width, channels = image.shape

# processed_image = pre_process(image)
# cards_list = get_contours(processed_image)
# idx = 0
# c = cards_list[idx].finish_card()
# print(c)
# cv2.drawContours(image, [c.outer_contours] + c.inner_contours, -1, (0, 255, 0), 5)

# cv2.imshow("Image with Con tours", image)
# # cv2.imshow("Image with Con tours", ic_image)


# end_time = time.time()
# execution_time = end_time - start_time
# print("Script execution time: {:.2f} seconds".format(execution_time))


# cv2.waitKey(0)
# cv2.destroyAllWindows()

