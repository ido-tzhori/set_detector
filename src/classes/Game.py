import cv2
import numpy as np
from . import ManyCards
from . import Card
import random
import time
import hashlib

font = cv2.FONT_HERSHEY_SIMPLEX

# Structure to store the information about a current game

class Game:
    def __init__(self, image):
        self.image = image  # input image of the game board
        self.sets_colors = {}
        self.thresh = 0
        self.BKG_THRESH = 217
        self.CARD_MAX_AREA = 50000
        self.CARD_MIN_AREA = 35000
        self.SHAPE_MIN_AREA = 3500
        self.cards = []
        self.sets = []

    def print_sets(self):
        """ Prints the content of each card in a set for debugging"""
        for set in self.sets:
            print(set[0][0], set[1][0],set[2][0])

    def pre_process(self):
        """ Grays, blurs, then chooses the threshold that picks up the most information.
            Can either be chosen adaptively by using the bkg_level variable or
            setting it manually through trial and error"""
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1)
        img_w, img_h = np.shape(self.image)[:2]
        bkg_level = gray[int(img_h/100)][int(img_w/2)]
        thresh_level = self.BKG_THRESH

        retval, thresh = cv2.threshold(blur,thresh_level,100,cv2.THRESH_BINARY)
        self.thresh = thresh
        return self
    
    def get_contours(self):
        """ Most important loop to pick up as much information on the first pass.
            In order to speed up calculation to get real time detection, it only passes
            through the contours once. Returns a list of cards based on the contour size.
            Starts filling in all information without calling helper methods"""
        contours, hierarchy = cv2.findContours(self.thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        card_list = []
        # store outer contour indices for comparison
        outer_indices = []
        # store current inner contours
        current_inner_contours = []
        current_inner_corner_points = []
        for i in range(len(contours)):
            size = cv2.contourArea(contours[i])
            if self.SHAPE_MIN_AREA < size < self.CARD_MAX_AREA: # relevant contours are bigger than minimum shape size
                                                                # and less than the maximum card size
                peri = cv2.arcLength(contours[i], True)
                approx = cv2.approxPolyDP(contours[i], 0.015 * peri, True) # stricter approx of polygon
                pts = np.float32(approx)

                # if the contour has no parent, it is an outer contour -> it is a card
                if hierarchy[0][i][3] == -1 and size > self.CARD_MIN_AREA:
                    if current_inner_contours:
                        # we add the inner contours to the previous card because we have moved to a new outer contour
                        card_list[-1].inner_contours += current_inner_contours
                        card_list[-1].inner_corner_points += current_inner_corner_points  # add inner contours' corner points to the last card
                        current_inner_contours = []  # reset the contours list
                        current_inner_corner_points = []

                    # initialize card if the condition hold and store all information gathered
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
        """ Simple loop to finish the classification of all cards"""
        classified_cards = []
        for c in self.cards:
            c.finish_card(self.image)
            classified_cards.append(c)
        self.cards = classified_cards
        return self

    def find_sets(self):
        """ Initializes a ManyCards class for simple storage of sets and cards
            returns a list of list (3 elements) of tuples where each tuple is a card in a set
            and the number of times that card has been seen in each game"""
        
        cards = ManyCards.ManyCards(self.cards)
        cards.return_all_sets().multiple()
        self.sets = cards.sets
        return self
    
    # def update_old_sets(self, current_sets):
    #     self.old_sets = current_sets

    #     old_sets = list(self.sets_colors.keys())
    #     for old_set in old_sets:
    #         if old_set not in [tuple(sorted([tup[0].id for tup in s])) for s in self.sets]:
    #             # this set doesn't exist anymore, so release its color
    #             self.display_colors[self.sets_colors[old_set]] = 0
    #             del self.sets_colors[old_set]
    #     return self
    
    def display_cards(self):
        """ Method to draw the information of a card on the card for debugging"""
        line_height = 45  # adjust this value as needed
        font = cv2.FONT_HERSHEY_SIMPLEX
        for card in self.cards:
            lines = [
                f"{card.shape}",
                 f"{str(card.count)}",
                f"{card.color}",
                 f"{card.shade}",
                # f"{card.center}",
                # f"{card.top_left}",
                # f"{card.bottom_right}",
                f"{round(card.avg_intensity, 5)}",
                # f"{card.id}"
            ]
            # adjust the x, y of the text
            x = card.center[0] - 70
            y = card.center[1] - 100
            for i, line in enumerate(lines):
                cv2.putText(self.image, line, (x, y + i * line_height), font, 1, (0,0,0), 4, cv2.LINE_AA)
            
            # cv2.drawContours(self.image, [card.inner_contours[0]], -1, (0, 255, 0), 2)

        return self.image

    def display_sets(self):
        """ Uses the list of list of tuples from the find_sets method to draw a border on the
            the three cards that make a set. Uses the number of times the card has seen to increment
            the border width and height"""
        for i, s in enumerate(self.sets):

            set_id = tuple(sorted([tup[0].id for tup in s]))  # need a hashable type to use as a dict key

            # use set_id as seed for random color generation. This ensures the color remains consistent
            # on every frame of the game
            seed = int(hashlib.sha256(str(set_id).encode('utf-8')).hexdigest(), 16) % 10**9
            random.seed(seed)

            # generate a unique color for this set
            set_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            self.sets_colors[set_id] = set_color  # store the color used by this set

            for tup in s:
                card = tup[0]
                m = tup[1]
                adj = 10
                top_left = card.top_left
                bottom_right = card.bottom_right
                
                # increments the width and height of the border by m (number of times the card has been used in a set)
                cv2.rectangle(self.image, tuple(map(lambda x: x - m * adj, top_left)),
                                            tuple(map(lambda x: x + m * adj, bottom_right)), set_color, adj)

        return self.image
