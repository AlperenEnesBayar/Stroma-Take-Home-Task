from utils import videoExtractor
from glob import glob


video_paths = glob("data\\images\\*\\*.mp4")

for video_path in video_paths:
    folder_in_path = video_path.split('\\')[:-1]
    videoExtractor(video_path, folder_in_path[0] + "\\" + folder_in_path[1] + "\\" + folder_in_path[2])
