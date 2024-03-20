import cv2
import os

def create_video_from_frames(image_folder, video_name):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    print(images[0])
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 10, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()
    cv2.destroyAllWindows()

# Example usage:
image_folder = "D:/bme/symp_frames2/"
video_name = "symp_tracker_demo2.mp4"
create_video_from_frames(image_folder, video_name)
