import numpy as np
import cv2
from collections import defaultdict, Counter
from . import utils

class Card:
    def __init__(self):
        self.count = 0
        self.shape = ''
        self.shade = ''
        self.color = ''
        self.outer_contours = []
        self.inner_contours = []
        self.corner_points = []
        self.inner_corner_points = []
        self.card_area = 0
        self.center = (0, 0)
        self.top_left = (0, 0)
        self.bottom_right = (0, 0)
        self.dominant_gbr = (0,0,0)
        self.avg_intensity = 0
        self.id = 0

    def __str__(self):
        return f"Card({self.count}, {self.shape}, {self.shade}, {self.color}"
    
    def make_id(self):
        self.id = int("".join([str(len(self.shape)),str(len(self.shade)),str(len(self.color)),str(self.count)]))
        return self
    
    def finish_card(self, image):

        self.count = len(self.inner_contours)
        self.cluster_pixels(image)
        color = utils.bgr_to_color(self.dominant_gbr)
        self.color = color
        self.get_shape()
        self.get_shading(image)
        self.calculate_center()
        self.make_id()

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
        if self.count != 0:
            first_contour = self.inner_contours[0]
            first_inner_corner_points = self.inner_corner_points[0]
            convexHull = cv2.convexHull(first_contour, returnPoints = False)
            convexityDefects = cv2.convexityDefects(first_contour, convexHull)
            max_defect_length = np.max([d[0][3] for d in convexityDefects])

            if len(first_inner_corner_points) == 4:
                shape = 'diamond'
            elif max_defect_length > 2000:
                shape = 'squiggle'
            else:
                shape = 'oval'
            
            self.shape = shape
            return self
    
    # def cluster_pixels_k_means(self, image, batch_size=1000):
    #     first_contour = self.inner_contours[0]
    #     x,y,w,h = cv2.boundingRect(first_contour)
    #     rect_image = image[y:y+h, x:x+w]
    #     all_colors = rect_image.reshape((rect_image.shape[0] * rect_image.shape[1], 3))

    #     # reduce colors using quantization
    #     all_colors = (all_colors // 64) * 64

    #     # use MiniBatchKMeans instead of regular KMeans
    #     clt = MiniBatchKMeans(n_clusters=2, batch_size=batch_size, n_init = 'auto')
    #     labels = clt.fit_predict(all_colors)
    #     label_counts = Counter(labels)
    #     first_dominant = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    #     second_dominant = clt.cluster_centers_[label_counts.most_common(2)[1][0]]

    #     if np.sum(first_dominant) > 400:
    #         self.dominant_gbr = second_dominant
    #     else:
    #         self.dominant_gbr = first_dominant

    #     density_ratio = list(label_counts.values())[0]/list(label_counts.values())[1]
    #     self.density_ratio = density_ratio

    #     return self

    def cluster_pixels(self, image, resize_dim=30):
        if self.count != 0:

            first_contour = self.inner_contours[0]
            x, y, w, h = cv2.boundingRect(first_contour)

            # resize the image to reduce the amount of data being processed
            rect_image = cv2.resize(image[y:y+h, x:x+w], (resize_dim, resize_dim))

            all_colors = rect_image.reshape(-1, 3)

            # reduce colors using quantization
            all_colors = (all_colors // 64) * 64

            # convert to list of tuples

            all_colors = [tuple(color) for color in all_colors]
            color_counts = Counter(all_colors)
            valid_colors = {color: count for color, count in color_counts.items() if sum(color) < 500 and
                            sum(color) != 0 and len(np.unique(color)) > 1}
            max_color = max(valid_colors, key=valid_colors.get)

            self.dominant_gbr = max_color

        return self

    def get_shading(self, image):
        if self.count != 0:

            x, y, w, h = cv2.boundingRect(self.inner_contours[0])

            roi = image[y:y+h, x:x+w]
            all_colors = roi.reshape(-1, 3)
            
            # Get all pixels where the sum of the pixel intensities is greater than 350
            high_intensity_pixels = np.where(np.sum(all_colors, axis=1) > 300)

            # Calculate the total number of pixels
            total_pixels = roi.shape[0] * roi.shape[1]

            # Calculate the ratio of high-intensity pixels
            avg_intensity = len(high_intensity_pixels[0]) / total_pixels

            # Find the two most common intensity values
            self.avg_intensity = avg_intensity

            if self.shape == 'squiggle':
                if avg_intensity < 0.4:
                    shade = 'full'
                elif avg_intensity < 0.84:
                    shade = 'striped'
                else:
                    shade = 'empty'
            elif self.shape == 'oval':
                if avg_intensity < 0.3:
                    shade = 'full'
                elif avg_intensity < 0.78:
                    shade = 'striped'
                else:
                    shade = 'empty'
            else:
                if avg_intensity < 0.6:
                    shade = 'full'
                elif avg_intensity < 0.84:
                    shade = 'striped'
                else:
                    shade = 'empty'

            self.shade = shade

            return self