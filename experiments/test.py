# +
import cv2
import numpy as np
import sklearn.cluster
from collections import Counter
from itertools import combinations
import random 

font = cv2.FONT_HERSHEY_COMPLEX

# +
class Card:
    def __init__(self, tl_br, color, shape, count, shade):
        self.tl_br = tl_br
        self.color = color
        self.shape = shape
        self.count = count
        self.shade = shade
        
    def __str__(self):
        return f"{self.count} {self.color} {self.shape} {self.shade}"
    
    
    def __hash__(self):
        return hash((self.shape, self.shade, self.count, self.color, self.tl_br))

    def __eq__(self, other):
        if not isinstance(other, SetCard):
            return False
        return self.shape == other.shape and self.shade == other.shade and self.count == other.count and self.color == other.color and self.tl_br == other.tl_br
    
def resize(img, scale):
    scale_percent = scale  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img


def pre_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
#     t_lower = 50  # Lower Threshold
#     t_upper = 150  # Upper threshold
#     edge = cv2.Canny(blur, t_lower, t_upper)

    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return ctrs


def relevant_contours(contours):
    cntrs = []
    corner_points = []
    bounding_rects = []
    max_area = np.max([cv2.contourArea(c) for c in contours])

    for cnt in contours:
        if max_area * 0.7 < cv2.contourArea(cnt) < max_area * 1.3:
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            pts = np.float32(approx)
            x, y, w, h = cv2.boundingRect(cnt)
            cntrs.append(cnt)
            corner_points.append(pts)
            bounding_rects.append([x,y,w,h])

    return cntrs, corner_points, bounding_rects



def birds_eye_view(img, corner_points, w , h):
    temp_rect = np.zeros((4, 2), dtype="float32")

    s = np.sum(corner_points, axis=2)

    tl = corner_points[np.argmin(s)]
    br = corner_points[np.argmax(s)]

    diff = np.diff(corner_points, axis=-1)
    tr = corner_points[np.argmin(diff)]
    bl = corner_points[np.argmax(diff)]

    if w <= 0.8 * h:  # If card is vertically oriented
        tl_tr_br_bl = np.array([tl, tr, br, bl])

    if w >= 1.2 * h:  # If card is horizontally oriented
        tl_tr_br_bl = np.array([bl, tl, tr, br])

    else:
        tl_tr_br_bl = np.array([tl, tr, br, bl])

    max_width = 200
    max_height = 300

    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], np.float32)
    M = cv2.getPerspectiveTransform(tl_tr_br_bl, dst)
    warped_image = cv2.warpPerspective(img, M, (max_width, max_height))

    return warped_image


def cards_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    t_lower = 50  # Lower Threshold
    t_upper = 150  # Upper threshold
    edge = cv2.Canny(blur, t_lower, t_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dilated = cv2.dilate(edge, kernel)
    _, thresh = cv2.threshold(dilated, 100, 255, cv2.THRESH_OTSU)
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in ctrs]
    max_area = np.max(areas)
    correct_ctrs = []
    for c in ctrs:
        if 0.7 * max_area < cv2.contourArea(c) < 1.3 * max_area:
            correct_ctrs.append(c)
    return correct_ctrs


def return_shape_count(img, correct_ctrs):
    count = len(correct_ctrs)

    first_contour = correct_ctrs[0]

    
    approx = cv2.approxPolyDP(first_contour, 0.01 * cv2.arcLength(first_contour, True), True)

    convexHull = cv2.convexHull(first_contour, returnPoints = False)
    convexityDefects = cv2.convexityDefects(first_contour, convexHull)
    max_defect_length = np.max([d[0][3] for d in convexityDefects])

    if len(approx) == 4:
        shape = 'diamonds'
    elif max_defect_length > 2000:
        shape = 'squiggle'
    else:
        shape = 'oval'
#     print(f'{count} shape is: {shape}')
    return shape, count, first_contour


def return_color_shade(image, first_contour, shape):
    x,y,w,h = cv2.boundingRect(first_contour)
    rect_image = image[y:y+h, x:x+w]
    rect_image = rect_image.reshape((rect_image.shape[0] * rect_image.shape[1], 3))

    clt = sklearn.cluster.KMeans(n_clusters=2)
    labels = clt.fit_predict(rect_image)
    label_counts = Counter(labels)
    max_count = label_counts[0]
    density_ratio = list(label_counts.values())[0]/list(label_counts.values())[1]
    print(density_ratio, label_counts)
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
        elif density_ratio < 6:
            shade = 'striped'
        else:
            shade = 'empty'

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # cluster and assign labels to the pixels
    clt = sklearn.cluster.KMeans(n_clusters=2)
    labels = clt.fit_predict(image)
    
    
    # count labels to find most popular
    label_counts = Counter(labels)
    
    # subset out most popular centroid
    second_dominant = clt.cluster_centers_[label_counts.most_common(2)[1][0]]
#     print(second_dominant)
    color = bgr_to_color(second_dominant)
    return color, shade


def bgr_to_color(bgr):
    # unpack the RGB values into separate variables
    b,g,r = bgr

    # if the red value is greater than the green and blue values, return red
    if r > g and r > b and r > 150:
        return "red"

    # if the green value is greater than the red and blue values, return green
    elif g > r and g > b:
        return "green"

    # if none of the above conditions are met, return purple
    else:
        return "purple"


def label_card(warped_image):
    correct_contours = cards_contours(warped_image)
    shape, count, first_contour = return_shape_count(warped_image, correct_contours)
    color, shade = return_color_shade(warped_image, first_contour, shape)
    return shape,count,color,shade


def is_set(card1, card2, card3):
    # Check if the color is a set
    if (card1.color == card2.color and card2.color == card3.color) or \
       (card1.color != card2.color and card2.color != card3.color and card1.color != card3.color):
        # Check if the number of shapes is a set
        if (card1.count == card2.count and card2.count == card3.count) or \
           (card1.count != card2.count and card2.count != card3.count and card1.count != card3.count):
            # Check if the shade is a set
            if (card1.shade == card2.shade and card2.shade == card3.shade) or \
               (card1.shade != card2.shade and card2.shade != card3.shade and card1.shade != card3.shade):
                # Check if the shape is a set
                if (card1.shape == card2.shape and card2.shape == card3.shape) or \
                   (card1.shape != card2.shape and card2.shape != card3.shape and card1.shape != card3.shape):
                    return True
    return False


def return_all_sets(cards):
    combos = list(combinations(cards, 3))
    sets = []
    for combo in combos:
        if is_set(*combo):
            sets.append(list(combo))
    return sets


def multiple(sets):
    r = []
    seen = {}
    for s in sets:
        tup_set = []
        for card in s:
            if card in seen:
                tup_set.append((card, seen[card] + 1))
                seen[card] += 1
            else:
                tup_set.append((card, 0))
                seen[card] = 0
        r.append(tup_set)
    return r


# +
colors = [
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (255, 255, 0), # Yellow
    (128, 0, 128), # Purple
    (0, 255, 255), # Cyan
    (255, 165, 0)  # Orange
]

img = cv2.imread('images/12_2.jpg') #diamond
img = resize(img, 30)
contours = pre_process(img)

cntrs, corner_points, bounding_rects = relevant_contours(contours)

cards = []

for i in range(len(cntrs)):
    x,y,w,h = bounding_rects[i]
    tl = (x,y)
    br = (x+w, y + h)
#     cv2.rectangle(img, tl, br, (255,0,0))
    warped_image = birds_eye_view(img, corner_points[i], bounding_rects[i][2], bounding_rects[i][3])
    shape,count,color,shade = label_card(warped_image)
    s = " ".join([shape,str(count),color,shade]) + ' ' + str(i)
    print(s)
#     cv2.putText(img, s, (x + w//2 - 50, y + h//2), font, 0.4, (0,255,0), 1)
    c = Card((tl, br), color, shape, count, shade)
    c.shape = shape
    cards.append(c)

print(len(cards))

sets = return_all_sets(cards)

tup_sets = multiple(sets)

for s in tup_sets:
    random_color = colors[random.randint(0, len(colors) - 1)]
    colors.remove(random_color)
    for tup in s:
        card = tup[0]
        m = tup[1]
        cv2.rectangle(img, tuple(map(lambda x: x - m * 20, card.tl_br[0])),
                                     tuple(map(lambda x: x + m * 20, card.tl_br[1])), random_color, 20)
cv2.imshow("x", img)

cv2.waitKey(0)
# -
sets

# +
# def nothing(x):
#     pass

# # cv2.namedWindow("Trackbar")
# # cv2.createTrackbar('l', "Trackbar", 25, 150, nothing)
# # cv2.createTrackbar('u', "Trackbar",100, 255, nothing)

# # while True:
# img = cv2.imread('images/12.jpg')
# img = resize(img, 30)
# # t1 = cv2.getTrackbarPos("l", "Trackbar")
# # t2 = cv2.getTrackbarPos("u", "Trackbar")
# contours = pre_process(img)
# cntrs, corner_points, bounding_rects = relevant_contours(contours)
# cv2.drawContours(img, contours, -1, (0,255,0),3)
# cv2.drawContours(img, cntrs, -1, (0,0,255),1)
# cv2.imshow("x", img)
# cv2.waitKey(0)
