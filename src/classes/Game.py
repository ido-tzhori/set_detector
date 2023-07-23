import cv2
import numpy as np
from . import ManyCards
from. import Card
import random
import time

font = cv2.FONT_HERSHEY_SIMPLEX

class Game:
    def __init__(self, image):
        self.image = image  # input image of the game board
        self.display_colors = {
                                (255, 0, 0):0,   # Red
                                (0, 255, 0):0,   # Green
                                (0, 0, 255):0,   # Blue
                                (255, 255, 0):0, # Yellow
                                (128, 0, 128):0, # Purple
                                (0, 255, 255):0, # Cyan
                                (255, 165, 0):0,  # Orange
                                (255, 192, 203):0, # Pink
                                (255, 255, 255):0, # White
                                (0, 0, 0):0,       # Black
                                (128, 128, 128):0, # Gray
                                (173, 216, 230):0, # Light Blue
                                }
        self.old_sets = None
        self.sets_colors = {}
        self.thresh = 0
        self.BKG_THRESH = 100
        self.CARD_MAX_AREA = 60000
        self.CARD_MIN_AREA = 35000
        self.SHAPE_MIN_AREA = 2000
        self.cards = []
        self.sets = []

    def pre_process(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1)
        img_w, img_h = np.shape(self.image)[:2]
        bkg_level = gray[int(img_h/100)][int(img_w/2)]
        thresh_level = bkg_level + self.BKG_THRESH

        retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)
        self.thresh = thresh
        return self
    
    def get_contours(self):
        contours, hierarchy = cv2.findContours(self.thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        card_list = []
        # store outer contour indices for comparison
        outer_indices = []
        # store current inner contours
        current_inner_contours = []
        current_inner_corner_points = []
        for i in range(len(contours)):
            size = cv2.contourArea(contours[i])
            if self.SHAPE_MIN_AREA < size < self.CARD_MAX_AREA:
                peri = cv2.arcLength(contours[i], True)
                approx = cv2.approxPolyDP(contours[i], 0.01 * peri, True)
                pts = np.float32(approx)
                # if the contour has no parent, it is an outer contour
                if hierarchy[0][i][3] == -1 and size > self.CARD_MIN_AREA:
                    if current_inner_contours:
                        # we add the inner contours to the previous card because we have moved to a new outer contour
                        card_list[-1].inner_contours += current_inner_contours
                        card_list[-1].inner_corner_points += current_inner_corner_points  # Add inner contours' corner points to the last card
                        current_inner_contours = []  # reset the inner contours list
                        current_inner_corner_points = []

                    # initialize card

                    card = Card.Card()
                    card.card_area = size
                    card.corner_points = pts
                    card.outer_contours = contours[i]
                    outer_indices.append(i)
                    card_list.append(card)
                elif hierarchy[0][i][3] in outer_indices:  # it's an inner contour if its parent is an outer contour
                    current_inner_contours.append(contours[i])
                    current_inner_corner_points.append(pts)

        # adding the inner contours for the last card after the loop ends
        if current_inner_contours:
            card_list[-1].inner_contours += current_inner_contours
            card_list[-1].inner_corner_points += current_inner_corner_points  # add inner contours' corner points to the last card
        
        self.cards = card_list
        return self

    def classify_all_cards(self):
        classified_cards = []
        for c in self.cards:
            c.finish_card(self.image)
            classified_cards.append(c)
        self.cards = classified_cards
        return self

    def find_sets(self):
        cards = ManyCards.ManyCards(self.cards)
        cards.return_all_sets().multiple()
        self.sets = cards.sets

        return self
    
    def update_old_sets(self, current_sets):
        self.old_sets = current_sets

        old_sets = list(self.sets_colors.keys())
        for old_set in old_sets:
            if old_set not in [tuple(sorted([tup[0].id for tup in s])) for s in self.sets]:
                # this set doesn't exist anymore, so release its color
                self.display_colors[self.sets_colors[old_set]] = 0
                del self.sets_colors[old_set]
        return self
    
    def display_cards(self):
        line_height = 60  # adjust this value as needed
        font = cv2.FONT_HERSHEY_SIMPLEX
        for card in self.cards:
            lines = [
                f"{card.shape}, {str(card.count)}",
                f"{card.color}, {card.shade}",
                f"{np.round(card.dominant_gbr)}",
                f"{round(card.avg_intensity, 5)}",
                # f"{card.id}"
            ]
            # adjust the x, y of the text
            x = card.center[0] - 70
            y = card.center[1] - 100
            for i, line in enumerate(lines):
                cv2.putText(self.image, line, (x, y + i * line_height), font, 1, (0,0,0), 4, cv2.LINE_AA)

        return self.image

    def display_sets(self):

        for i, s in enumerate(self.sets):

            set_id = tuple(sorted([tup[0].id for tup in s]))  # need a hashable type to use as a dict key
            if set_id in self.sets_colors:
                # this set was present in the last run, so just reuse its color
                set_color = self.sets_colors[set_id]
            else:
                # this set is new, so assign it an unused color
                # (use the while loop from before to find an unused color)
                set_hash = abs(hash(set_id)) % len(self.display_colors)
                while list(self.display_colors.values())[set_hash]:
                    set_hash = (set_hash + 1) % len(self.display_colors)

                set_color = list(self.display_colors.keys())[set_hash]
                self.display_colors[set_color] = 1
                self.sets_colors[set_id] = set_color  # store the color used by this set

            for tup in s:
                card = tup[0]
                m = tup[1]
                adj = 50  
                cv2.rectangle(self.image, tuple(map(lambda x: x - m * adj, card.top_left)),
                                            tuple(map(lambda x: x + m * adj, card.bottom_right)), set_color, adj)
        # if self.old_sets:
        #     for old_set in self.old_sets:
        #         if old_set not in [tuple(sorted([tup[0].id for tup in s])) for s in self.sets]:
        #             # this set doesn't exist anymore, so release its color
        #             self.display_colors[self.sets_colors[old_set]] = 0
        #             del self.sets_colors[old_set]

        return self.image
