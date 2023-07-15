import numpy as np
import cv2
from joblib import Parallel, delayed
import time

start_time = time.time()


image = cv2.imread("images/12.jpg")

height, width, channels = image.shape

def pre_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (31, 31), 5)
    return blur

def get_contours(img):
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(ctrs, key=cv2.contourArea, reverse=True)
    max_contour_area = cv2.contourArea(sorted_contours[0])
    return sorted_contours, max_contour_area

def relevant_contour(cnt, max_area, epsilon_value):
    if max_area * 0.7 < cv2.contourArea(cnt) < max_area * 1.3:
        approx = cv2.approxPolyDP(cnt, epsilon_value * cv2.arcLength(cnt, True), True)
        pts = np.float32(approx)
        x, y, w, h = cv2.boundingRect(cnt)
        return cnt, pts, np.array([x,y,w,h])
    else:
        pass

def relevant_contours_parallel(contours, max_area_threshold, epsilon_value):
    def process_contour(cnt):
        area = cv2.contourArea(cnt)
        if max_area_threshold * 0.7 < area < max_area_threshold * 1.3:
            approx = cv2.approxPolyDP(cnt, epsilon_value * cv2.arcLength(cnt, True), True)
            pts = np.float32(approx)
            x, y, w, h = cv2.boundingRect(cnt)
            return cnt, pts, np.array([x, y, w, h])
        else:
            return None

    # Process contours in parallel and filter relevant contours
    relevant_contours = [cnt for cnt in Parallel(n_jobs=-1)(delayed(process_contour)(cnt) for cnt in contours) if cnt is not None]

    return relevant_contours

processed_image = pre_process(image)
contours, max_area_threshold = get_contours(processed_image)
epsilon_value = 0.009
relevant_contours = relevant_contours_parallel(contours, max_area_threshold, epsilon_value)

cv2.drawContours(image, [contour for contour, _, _ in relevant_contours], -1, (0, 255, 0), 7)
cv2.imshow("Image with Contours", image)

end_time = time.time()
execution_time = end_time - start_time
print("Script execution time: {:.2f} seconds".format(execution_time))


cv2.waitKey(0)
cv2.destroyAllWindows()

