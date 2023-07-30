import cv2
from classes import Game
import time
from classes import utils
import numpy as np

def process_frame(frame, canvas_width=400):
    """
    Processes a single frame of the video to identify and count card sets.

    This function creates a Game instance for the given frame, performs pre-processing,
    contour detection, card classification, and set finding on the frame. It then displays
    the identified cards and sets on the frame, and displays the warped card sets on a canvas.
    """
    g = Game.Game(frame)
    g.pre_process()
    g.get_contours()
    g.classify_all_cards()
    g.find_sets()
    g.sort_sets()
    frame_height, _, _ = frame.shape
    canvas = g.display_warps(canvas_width, frame_height)

    g.display_cards()  # uncomment if you want to see the information as text

    # concatenate the video frame and the canvas
    processed_frame = np.concatenate((g.display_sets(), canvas), axis=1)

    return processed_frame, len(g.sets)


def main():
    """
    Main function to start the processing of the video.

    It reads a video file, processes each frame to identify card sets, adds the elapsed time 
    and number of sets to the frame, and displays the updated frame. It continues this process 
    until all frames have been read or the user presses 'q'. It then releases the video file 
    and destroys all created windows, and prints the total elapsed time.
    """
    start_time = time.time()
    path = '/Users/idotzhori/Desktop/set_detector/videos/video_3.MOV'
    video = cv2.VideoCapture(path)

    ret, frame = video.read()  # read the first frame

    if not ret:
        print("Error: Can't read video")
        return

    # process and manipulate the first frame
    frame, _ = process_frame(frame)

    # Define the codec and create a VideoWriter object with the dimensions of the processed frame
    frame_height, frame_width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

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
        font_scale = 2
        color = (255, 255, 255)  # white color
        thickness = 4

        # get the size of the text box
        text_size, _ = cv2.getTextSize(text_sets, font, font_scale, thickness)

        # get the width and height of the video frame
        frame_height, frame_width, _ = frame.shape
        
        # calculate the position of the text (centered on top)
        position = (400, 60) 

        # add text to the frame
        cv2.putText(frame, text_sets, position, font, font_scale, color, thickness)

        # write the frame to the output video file
        out.write(frame)

        # show the frame
        cv2.imshow('Video feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
            break

    video.release()
    out.release()  # release the VideoWriter
    cv2.destroyAllWindows()
    print('Done')

if __name__ == "__main__":
    main()