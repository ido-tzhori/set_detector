import cv2
from classes import Game
import time

def process_frame(frame):
    g = Game.Game(frame)
    g.pre_process()
    g.get_contours()
    g.classify_all_cards()
    print(len(g.cards))
    g.find_sets()
    # processed_frame = g.display_cards()
    processed_frame = g.display_sets()
    g.update_old_sets(g.sets_colors)
    return processed_frame

def main():
    video = cv2.VideoCapture('/Users/idotzhori/Desktop/set_detector/images/video.MOV')

    # get the start time
    start_time = time.time()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # calculate the elapsed time in seconds
        elapsed_time = time.time() - start_time

        # convert the elapsed time to a formatted string
        formatted_time = time.strftime('%S', time.gmtime(elapsed_time))

        # add the elapsed time to the frame
        text = f'time Elapsed: {formatted_time}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)  # white color
        thickness = 2
        position = (10, 30)  # top-left corner

        frame = process_frame(frame)

        cv2.putText(frame, text, position, font, font_scale, color, thickness)
        cv2.imshow('Video feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
            break

    video.release()
    cv2.destroyAllWindows()

    print(elapsed_time)

if __name__ == "__main__":
    main()