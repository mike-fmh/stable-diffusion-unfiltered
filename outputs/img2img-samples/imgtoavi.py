import cv2
import os

image_folder = r"E:\Documents\Github\stable-diffusion-unfiltered\outputs\img2img-samples\heat100"
video_name = image_folder.split("\\")[-1] + '.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 60, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
