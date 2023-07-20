import cv2
import numpy as np
from Card import Card
from ManyCards import ManyCards
import random

colors = [
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (255, 255, 0), # Yellow
    (128, 0, 128), # Purple
    (0, 255, 255), # Cyan
    (255, 165, 0),  # Orange
    (255, 192, 203), # Pink
    (255, 255, 255), # White
    (0, 0, 0),       # Black
    (128, 128, 128), # Gray
    (165, 42, 42),   # Brown
    (255, 255, 224), # Light Yellow
    (173, 216, 230), # Light Blue
]

font = cv2.FONT_HERSHEY_SIMPLEX


class Game:
    def __init__(self, image):
        self.image = image  # input image of the game board
        self.thresh = 0
        self.BKG_THRESH = 30
        self.CARD_MAX_AREA = 400000
        self.CARD_MIN_AREA = 200000
        self.SHAPE_MIN_AREA = 15000
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
        card_count = 0 ######## ADD COUNTER TO STOP EVENTUALLY #####

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

                    card = Card()
                    card.card_area = size
                    card.corner_points = pts
                    card.outer_contours = contours[i]
                    outer_indices.append(i)
                    card_list.append(card)
                    card_count += 1
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

        classified_cards = [c.finish_card(image) for c in self.cards]
        self.cards = classified_cards

        return self

    def find_sets(self):

        cards = ManyCards(self.cards)
        cards.return_all_sets().multiple()
        print(cards)
        self.sets = cards.sets

        return self

    def display_cards(self):
        line_height = 60  # Adjust this value as needed
        font = cv2.FONT_HERSHEY_SIMPLEX
        for card in self.cards:
            lines = [
                f"{card.shape}, {str(card.count)}",
                f"{card.color}, {card.shade}",
            ]
            x = card.center[0] - 200
            y = card.center[1] + 100
            for i, line in enumerate(lines):
                cv2.putText(self.image, line, (x, y + i * line_height), font, 2, (255,255,255), 10, cv2.LINE_AA)


    def display_sets(self):
        cv2.namedWindow("x")

        for s in self.sets:
            random_color = colors[random.randint(0, len(colors) - 1)]
            colors.remove(random_color)
            for tup in s:
                card = tup[0]
                m = tup[1]
                adj = 20
                cv2.rectangle(self.image, tuple(map(lambda x: x - m * adj, card.top_left)),
                                            tuple(map(lambda x: x + m * adj, card.bottom_right)), random_color, adj)
                
        cv2.imshow("Image with Con tours", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

image = cv2.imread("images/15.jpg")
g = Game(image)

g.pre_process().get_contours().classify_all_cards()
g.find_sets()
g.display_cards()
g.display_sets()
# for i in g.sets:
#     for c in i:
#         print(c)
# cardd = g.cards[7]
# cv2.drawContours(image, [cardd.outer_contours] + cardd.inner_contours, -1, (0, 255, 0), 5)

# cv2.imshow("Image with Con tours", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()