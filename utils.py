import os
import cv2
from glob import glob


# Generates a name based on the image's number.
def fileNameGenerator(counter, name_lenght=4, extension="jpg"):
    image_name = ""
    for _ in range(name_lenght - len(str(counter))):
        image_name += '0'
    image_name += str(counter)
    if len(image_name) > name_lenght:
        print("Warning: The size of the name of the iamge exceeded the maximum size.")
    image_name += ('.' + extension)
    return image_name


# Reads the video and divides it into images
def videoExtractor(video_path, output_folder_path):
    # Checking the video path if exists read it with OpenCV
    if os.path.exists(video_path):
        vidcap = cv2.VideoCapture(video_path)
    else:
        raise "Image not found. Please check the path"

    # If output folder not exists create a new directory
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(output_folder_path + "/" + fileNameGenerator(count), image)
        success, image = vidcap.read()
        count += 1
    print(f'Total of {count + 1} images is saved.')


# Finds all the videos in the specified data file and extracts their images. Then it deletes the videos.
def extractAllVideos():
    video_paths = glob("data\\images\\*\\*.mp4")
    for video_path in video_paths:
        folder_in_path = video_path.split('\\')[:-1]
        videoExtractor(video_path, folder_in_path[0] + "\\" + folder_in_path[1] + "\\" + folder_in_path[2])
        os.remove(video_path)
