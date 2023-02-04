import os
import cv2
import json
from glob import glob


# Generates a name based on the image's id.
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
        raise "Video not found. Please check the path"

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
def extractAllVideos(dataFileSystem):  # Ex: "data\\images\\*\\*.mp4"
    video_paths = glob(dataFileSystem)
    for video_path in video_paths:
        folder_in_path = video_path.split('\\')[:-1]
        videoExtractor(video_path, folder_in_path[0] + "\\" + folder_in_path[1] + "\\" + folder_in_path[2])
        os.remove(video_path)


def instance2Yolov7Label(label_path, output_folder_path, w_img=640, h_img=640):
    if os.path.exists(label_path):
        f = open(label_path)
        data = json.load(f)
        f.close()  # Closing file

        # if it doesn’t exist we create one
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        # Iterating through the json
        for single_annotation in data['annotations']:
            text_file_name = fileNameGenerator(single_annotation['image_id'] - 1, extension='txt')


            f = open(output_folder_path + "\\" + text_file_name, "a")
            id = '0' if single_annotation['category_id'] == 1 else '1'
            x = int(single_annotation['bbox'][0])
            y = int(single_annotation['bbox'][1])
            w = int(single_annotation['bbox'][2])
            h = int(single_annotation['bbox'][3])


            xcenter = (x + (w / 2))
            ycenter = (y + (h / 2))
            xcenter /= w_img
            ycenter /= w_img
            w /= w_img
            h /= w_img

            f.write('{} {:.6} {:.6} {:.6} {:.6}\n'.format(id, xcenter, ycenter, w, h))
            # f.write(id + " " +
            #         str(xcenter) + " " +
            #         str(ycenter) + " " +
            #         str(w) + " " +
            #         str(h) + '\n')
            f.close()

        for file in glob(output_folder_path + "\\*.txt"):
            with open(file, 'r+') as fp:
                lines = fp.readlines()
                fp.seek(0)
                fp.truncate()
                lines[-1] = lines[-1].replace("\n", "")
                fp.writelines(lines)

    else:
        raise "Label not found. Please check the path"


def convertYolov7Labels():
    for pa in ['train', 'test', 'val']:
        instance2Yolov7Label("labels/instances_" + pa + ".json", "bnn_data/labels/" + pa)

convertYolov7Labels()
