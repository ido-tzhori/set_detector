import numpy as np
import cv2
from collections import defaultdict, Counter
from . import utils

# structure to hold information about a card used for set detection and debugging

class Card:
    def __init__(self):
        self.count = 0
        self.shape = ''
        self.shade = ''
        self.color = ''
        self.outer_contours = [] # larger parent contours of the inner contours
        self.inner_contours = [] # for the count (number of shapes)
        self.corner_points = [] # for card detection and border drawing
        self.inner_corner_points = [] # for shape detection
        self.card_area = 0 # size in pixels of the contour
        self.center = (0, 0)
        self.top_left = (0, 0)
        self.bottom_right = (0, 0)
        self.dominant_gbr = (0, 0, 0)
        self.avg_intensity = 0
        self.id = 0

    def __str__(self):
        """Returns a string representation of a card - used for debugging"""
        return f"Card({self.count}, {self.shape}, {self.shade}, {self.color})"
    
    def make_id(self):
        """Uses the features of the card to make a unique id for a card"""

        self.id = int("".join([str(len(self.shape)),str(len(self.shade)),str(len(self.color)),str(self.count)]))

        return self
    
    def finish_card(self, image):

        """Finalizes the card classification based on the outer and inner contours of a card"""
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
        """ Stores the center, top left, and bottom right corner of the card
            used for drawing of the set borders"""
        if len(self.outer_contours) > 0:
            x, y, w, h = cv2.boundingRect(self.outer_contours)
            self.center = (int(x + w/2), int(y + h/2))
            if w < 176: # sometime it bugs for some reason -> set it manually
                self.top_left = (x,y)
                self.bottom_right = (x + w, y + h)
            else:
                self.top_left = (x,y)
                self.bottom_right = (x + 172, y + h)
        else:
            self.center = (0, 0)  # reset to (0, 0) if no contours

        return self

    def get_shape(self):
        """Determines and assigns the shape of the card.

        The shape of the card is identified based on its convexity defects and inner corner points. If the number
        of inner corner points is 4, it is a diamond. If the maximum length of convexity defects is more than 1000, 
        it is a squiggle. Otherwise, it is an oval"""

        if self.count != 0:
            first_contour = self.inner_contours[0] # first contour is the contour of the shape
            first_inner_corner_points = self.inner_corner_points[0] # estimated cornor points of the shape
            if len(self.inner_corner_points) > 1:
                second_inner_corner_points = self.inner_corner_points[1] # double checks for clarity
            else:
                second_inner_corner_points = []
            convexHull = cv2.convexHull(first_contour, returnPoints = False)
            convexityDefects = cv2.convexityDefects(first_contour, convexHull)
            max_defect_length = np.max([d[0][3] for d in convexityDefects]) # max defect length of bounding rectangle
                                                                            # measured in pixels

            if len(first_inner_corner_points) == 4 or len(second_inner_corner_points) == 4:
                shape = 'diamond'
            elif max_defect_length > 1000:
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
        """ Shrinks the image for faster processing. Then it returns the dominant
            color that is not a 'dull' color based on criteria from trial and error"""
        if self.count != 0:

            first_contour = self.inner_contours[0]
            x, y, w, h = cv2.boundingRect(first_contour)

            # resize the image to reduce the amount of data being processed
            rect_image = cv2.resize(image[y:y+h, x:x+w], (resize_dim, resize_dim))

            all_colors = rect_image.reshape(-1, 3)

            # reduce colors using quantization -> less colors for Counter()
            all_colors = (all_colors // 64) * 64

            # convert to list of tuples

            all_colors = [tuple(color) for color in all_colors] # needs to be hashable
            color_counts = Counter(all_colors)
            valid_colors = {color: count for color, count in color_counts.items() if sum(color) < 600 and
                            sum(color) != 0 and len(np.unique(color)) > 1} # filters the white background of the card
            max_color = max(valid_colors, key=valid_colors.get)

            self.dominant_gbr = max_color

        return self

    def get_shading(self, image):
        """Determines and assigns the shading of the card.

        Calculates the average pixel intensity of pixels based on criteria found through
        trial and error.

        Based on the average intensity and the shape of the card, it then determines the shading of the card: 
        'full', 'striped', or 'empty'. These thresholds vary based on the card's shape"""

        if self.count != 0:

            x, y, w, h = cv2.boundingRect(self.inner_contours[0])

            roi = image[y:y+h, x:x+w]
            all_colors = roi.reshape(-1, 3)
            
            # get all pixels where the sum of the pixel intensities is greater than 350
            high_intensity_pixels = np.where(np.sum(all_colors, axis=1) > 650)

            # calculate the total number of pixels
            total_pixels = roi.shape[0] * roi.shape[1]

            # calculate the ratio of high-intensity pixels

            avg_intensity = len(high_intensity_pixels[0]) / total_pixels
            self.avg_intensity = avg_intensity

            # different values for different shapes
            if self.shape == 'squiggle':
                if avg_intensity < 0.4:
                    shade = 'full'
                elif avg_intensity < 0.78:
                    shade = 'striped'
                else:
                    shade = 'empty'
            elif self.shape == 'oval':
                if avg_intensity < 0.3:
                    shade = 'full'
                elif avg_intensity < 0.65:
                    shade = 'striped'
                else:
                    shade = 'empty'
            else:
                if avg_intensity < 0.55:
                    shade = 'full'
                elif avg_intensity < 0.82:
                    shade = 'striped'
                else:
                    shade = 'empty'

            self.shade = shade

            return self
    
    # def get_shading(self, image):
    #     if self.count != 0:

    #         x, y, w, h = cv2.boundingRect(self.inner_contours[0])

    #         roi = image[y:y+h, x:x+w]
    #         all_colors = roi.reshape(-1, 3)

    #         # Calculate variance for each channel
    #         variance_intensity = np.var(all_colors, axis=0)

    #         # Calculate mean intensity for each channel
    #         mean_intensity = np.mean(all_colors, axis=0)

    #         print(mean_intensity)

    #         # Classify shading based on intensity variance and mean intensity
    #         if np.all(variance_intensity < 30):
    #             if np.all(mean_intensity < 128):
    #                 shade = 'full'
    #             else:
    #                 shade = 'empty'
    #         else:
    #             shade = 'striped'

    #         self.shade = shade

    #     return self
