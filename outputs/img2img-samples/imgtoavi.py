import cv2
import os
import unicodedata
import re
import tqdm


INPUT_DIR = r"E:\Documents\shrek\frames\allstar"

# output dir will become input dir without the last directory
OUTPUT_DIR = ""
for i in range(len(INPUT_DIR)):
    OUTPUT_DIR += INPUT_DIR[i]
    found_slash = False
    for j in range(i, len(INPUT_DIR)):
        if INPUT_DIR[j] == "\\":
            found_slash = True
            break
    if found_slash:
        continue
    else:
        break
OUTPUT_DIR = OUTPUT_DIR[:-1]


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def bi_bubble_sort(array, array_to_follow):
    n = len(array)
    for i in range(n):
        # Create a flag that will allow the function to
        # terminate early if there's nothing left to sort
        already_sorted = True
        # Start looking at each item of the list one by one,
        # comparing it with its adjacent value. With each
        # iteration, the portion of the array that you look at
        # shrinks because the remaining items have already been
        # sorted.
        for j in range(n - i - 1):
            if array[j] > array[j + 1]:
                # If the item you're looking at is greater than its
                # adjacent value, then swap them
                array[j], array[j + 1] = array[j + 1], array[j]
                array_to_follow[j], array_to_follow[j+1] = array_to_follow[j+1], array_to_follow[j]
                # Since you had to swap two elements,
                # set the `already_sorted` flag to `False` so the
                # algorithm doesn't finish prematurely
                already_sorted = False
        # If there were no swaps during the last iteration,
        # the array is already sorted, and you can terminate
        if already_sorted:
            break


def convert_to_valid_filename(filename, extension):
    fileexists = True
    fname = filename
    i = 0
    while fileexists:
        i += 1
        use_fname = slugify(fname + f"-{i}")
        fileexists = os.path.isfile(use_fname + extension)
    return use_fname + extension


image_folder = INPUT_DIR
video_name = OUTPUT_DIR + convert_to_valid_filename(image_folder.split("\\")[-1], ".avi")
print(video_name)

images, imnames = [], []
for img in os.listdir(image_folder):
    images.append(img)
    try:
        # filename formatted like 0-1.png
        imnames.append(int(img.split("-")[0]))
    except:
        # filename formatted like 0.png
        imnames.append(int(img.split(".")[0]))
bi_bubble_sort(imnames, images)
print(images)
print(imnames)

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 30, (width, height))

for image in tqdm.tqdm(images):
    #   print(image)
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
