import numpy as np
import cv2
import sklearn.cluster
from sklearn.cluster import MiniBatchKMeans

import time
from collections import defaultdict, Counter
from utils import *

scale = 1
CARD_MAX_AREA = 400000 * scale
CARD_MIN_AREA = 200000 * scale
SHAPE_MIN_AREA = 15000 * scale

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
        self.dominant_gbr = (0,0,0)
        self.density_ratio = 0

    def __str__(self):
        return f"Card({self.count}, {self.shape}, {self.shade}, {self.color}"
    
    def finish_card(self, image):
        # image = resize(image, scale)

        self.count = len(self.inner_contours)
        self.cluster_pixels(image)
        self.get_shading()

        color = bgr_to_color(self.dominant_gbr)
        self.color = color
        self.get_shape()
        self.calculate_center()

        return self
    
    def calculate_center(self):

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
    
    def cluster_pixels_k_means(self, image, batch_size=1000):
        first_contour = self.inner_contours[0]
        x,y,w,h = cv2.boundingRect(first_contour)
        rect_image = image[y:y+h, x:x+w]
        all_colors = rect_image.reshape((rect_image.shape[0] * rect_image.shape[1], 3))

        # reduce colors using quantization
        all_colors = (all_colors // 64) * 64

        # use MiniBatchKMeans instead of regular KMeans
        clt = MiniBatchKMeans(n_clusters=2, batch_size=batch_size, n_init = 'auto')
        labels = clt.fit_predict(all_colors)
        label_counts = Counter(labels)
        first_dominant = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
        second_dominant = clt.cluster_centers_[label_counts.most_common(2)[1][0]]

        if np.sum(first_dominant) > 400:
            self.dominant_gbr = second_dominant
        else:
            self.dominant_gbr = first_dominant

        density_ratio = list(label_counts.values())[0]/list(label_counts.values())[1]
        self.density_ratio = density_ratio

        return self

    def cluster_pixels(self, image, resize_dim=40):
        first_contour = self.inner_contours[0]
        x, y, w, h = cv2.boundingRect(first_contour)

        # resize the image to reduce the amount of data being processed
        rect_image = cv2.resize(image[y:y+h, x:x+w], (resize_dim, resize_dim))

        all_colors = rect_image.reshape(-1, 3)

        # reduce colors using quantization
        all_colors = (all_colors // 64) * 64

        # convert to list of tuples
        all_colors = [tuple(color) for color in all_colors]

        # count color occurrences
        color_counts = Counter(all_colors)
        print(color_counts)
        # get dominant colors
        dominant_colors = color_counts.most_common(2)
        valid_colors = {color: count for color, count in color_counts.items() if sum(color) < 300 and
                         sum(color) != 0 and len(np.unique(color)) > 1}
        max_color = max(valid_colors, key=valid_colors.get)

        if np.sum(dominant_colors[0][0]) > 400:
            self.dominant_gbr = dominant_colors[1][0]
        else:
            self.dominant_gbr = dominant_colors[0][0]
        self.dominant_gbr = max_color
        # calculate density ratio
        density_ratio = dominant_colors[0][1] / dominant_colors[1][1]
        self.density_ratio = density_ratio

        return self

    def get_shading(self):
        density_ratio = self.density_ratio
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


