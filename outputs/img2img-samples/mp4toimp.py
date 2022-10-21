import cv2
import os
import tqdm

vidcap = cv2.VideoCapture(r"C:\Users\mmh\Downloads\Y2Mate.is - Glass Animals - Heat Waves (Official Video)-mRD0-GxqHVo-1080p-1656206039429.mp4")
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
success,image = vidcap.read()
count = 0
width, height = 1024, 574
storedir = f"heatwave_imgs_{width}x{height}"

if not os.path.exists(f"{storedir}"):
  os.makedirs(f"{storedir}")
for count in tqdm.trange(length):
  image = cv2.resize(image, (width, height))
  cv2.imwrite(f"{storedir}/%d.png" % count, image)
  success,image = vidcap.read()
  #print('Read a new frame: ', success)
