import cv2
from classes import Game
import time

def process_frame(frame):
    """
    Processes a single frame of the video to identify and count card sets.

    This function creates a Game instance for the given frame, performs pre-processing,
    contour detection, card classification, and set finding on the frame. It then displays
    the identified cards and sets on the frame."""
    g = Game.Game(frame)
    g.pre_process()
    g.get_contours()
    g.classify_all_cards()
    g.find_sets()
    # processed_frame = g.display_cards() # uncomment if you want to see the information as text
    processed_frame = g.display_sets()
    return processed_frame, len(g.sets)

def main():
    """
    Main function to start the processing of the video.

    It reads a video file, processes each frame to identify card sets, adds the elapsed time 
    and number of sets to the frame, and displays the updated frame. It continues this process 
    until all frames have been read or the user presses 'q'. It then releases the video file 
    and destroys all created windows, and prints the total elapsed time.
    """

    # add the path of the video
    video = cv2.VideoCapture('/Users/idotzhori/Desktop/set_detector/images/video_3.MOV')

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
        text_time = f'time elapsed: {formatted_time}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)  # white color
        thickness = 2
        position = (10, 30)  # top-left corner

        # process and manipulate the frame
        frame, n = process_frame(frame)

        cv2.putText(frame, text_time, position, font, font_scale, color, thickness)

        text_sets = f'number of sets: {n}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)  # white color
        thickness = 2

        # get the size of the text box
        text_size, _ = cv2.getTextSize(text_sets, font, font_scale, thickness)

        # get the width and height of the video frame
        frame_height, frame_width, _ = frame.shape
        
        # calculate the position of the text (centered on top)
        position = (frame_width//2 - text_size[0]//2, 30) 

        # show the frame
        cv2.putText(frame, text_sets, position, font, font_scale, color, thickness)
        cv2.imshow('Video feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
            break

    video.release()
    cv2.destroyAllWindows()

    print(elapsed_time)

if __name__ == "__main__":
    main()