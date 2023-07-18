
import numpy as np
import cv2
import sklearn.cluster
import time
from collections import defaultdict, Counter

CARD_MAX_AREA = 400000
CARD_MIN_AREA = 200000
SHAPE_MIN_AREA = 15000

BKG_THRESH = 30
# CARD_THRESH = 10000

class Card:
    def __init__(self):
        self.count = 0
        self.shape = 'unknown'
        self.shading = 'unknown'
        self.color = 'unknown'
        self.outer_contours = []
        self.inner_contours = []
        self.corner_points = []
        self.inner_corner_points = []
        self.card_area = 0

    def __str__(self):
        return f"Card({self.count}, {self.shape}, {self.shading}, {self.color},\
            {len(self.inner_contours)}, {self.card_area}"

start_time = time.time()

def pre_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    img_w, img_h = np.shape(img)[:2]
    bkg_level = gray[int(img_h/100)][int(img_w/2)]
    thresh_level = bkg_level + BKG_THRESH

    retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)
    return thresh

def get_contours(thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    card_list = []
    # store outer contour indices for comparison
    outer_indices = []
    # store current inner contours
    current_inner_contours = []
    current_inner_corner_points = []
    card_count = 0 ######## ADD COUNTER TO STOP EVENTUALLY #####

    for i in range(len(contours)):
        size = cv2.contourArea(contours[i])
        if SHAPE_MIN_AREA < size < CARD_MAX_AREA:
            peri = cv2.arcLength(contours[i], True)
            approx = cv2.approxPolyDP(contours[i], 0.01 * peri, True)
            pts = np.float32(approx)

            # if the contour has no parent, it is an outer contour
            if hierarchy[0][i][3] == -1 and size > CARD_MIN_AREA:
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
    return card_list

def preprocess_card(contour, corner_points, image):
    """uses the contour to identify the information about a card"""

    # initialize card

    x,y,w,h = cv2.boundingRect(contour)

    # average = np.sum(corner_points, axis=0)/len(corner_points)
    # cent_x = int(average[0][0])
    # cent_y = int(average[0][1])

    bird_i_view = flattener(image, corner_points, w, h)

    return bird_i_view

def finish_card(card: Card):
    count = len(card.inner_contours)
    card.count = count
    
    first_contour = card.inner_contours[0]
    color = dominant_color(first_contour)
    print(color)
    return first_contour


def dominant_color(first_contour):
    x,y,w,h = cv2.boundingRect(first_contour)
    rect_image = image[y:y+h, x:x+w]
    all_colors = rect_image.reshape((rect_image.shape[0] * rect_image.shape[1], 3))
    clt = sklearn.cluster.KMeans(n_clusters=2)
    labels = clt.fit_predict(all_colors)
    label_counts = Counter(labels)
    second_dominant = clt.cluster_centers_[label_counts.most_common(2)[1][0]]
    color = bgr_to_color(second_dominant)
    return color



def bgr_to_color(bgr):
    # unpack the RGB values into separate variables
    b,g,r = bgr

    # if the red value is greater than the green and blue values, return red
    if r > g and r > b and r > 180:
        return "red"

    # if the green value is greater than the red and blue values, return green
    elif g > r and g > b:
        return "green"

    # if none of the above conditions are met, return purple
    else:
        return "purple"
    
def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized, grayed image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.
    
    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
            
        
    maxWidth = 200
    maxHeight = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warp

image = cv2.imread("images/15.jpg")

height, width, channels = image.shape

processed_image = pre_process(image)
cards_list = get_contours(processed_image)
idx = 8
card_idx = cards_list[idx]
# print(cards_list[idx])
outer_contours = [card_idx.outer_contours]
inner_contours = card_idx.inner_contours
ic = finish_card(card_idx)
# ic_image = dominant_color(ic, image)
# birds_i_view = preprocess_card(contours[0], corner_points[0], image)

cv2.drawContours(image, inner_contours + outer_contours, -1, (0, 255, 0), 2)
# cv2.drawContours(image, ic, -1, (0, 255, 0), 5)

cv2.imshow("Image with Con tours", image)
# cv2.imshow("Image with Con tours", ic_image)


end_time = time.time()
execution_time = end_time - start_time
print("Script execution time: {:.2f} seconds".format(execution_time))


cv2.waitKey(0)
cv2.destroyAllWindows()

