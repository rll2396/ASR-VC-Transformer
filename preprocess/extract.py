import ast
import os

import csv
import cv2
import moviepy.editor
import numpy as np
import pandas as pd
from scipy.io import wavfile
from PIL import Image

"""
This script expects the HOW2 dataset's subtitle data ("text.id.en" and
"segments" files) to be under datasets/HOW2/subtitles/{split_name} and all
downloaded mp4s from Youtube to be under datasets/HOW2/vids
"""

split = "train"
for split in ("val", "dev5"):
    data_dir = "datasets/HOW2/"

    preprocess_path = f'{data_dir}preprocess.csv'
    with open(preprocess_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        header = ['videoid', 'segmentid', 'text']
        writer.writerow(header)

        # write the data
        with open(f'{data_dir}subtitles/{split}/text.id.en', 'r') as file1:
            Lines = file1.readlines()

        for line in Lines:
            videoid = line[:11]
            segmentnumber = line.split()[0].split('_')[-1]
            captions = line.split()[1:]
            data = [videoid, segmentnumber,captions]
            writer.writerow(data)
        file1.close()


    withtime_path = f'{data_dir}withtime.csv'
    with open(withtime_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        header = ['videoid', 'segmentid', 'timestamp']
        writer.writerow(header)

        with open(f'{data_dir}subtitles/{split}/segments', 'r') as file1:
            Lines = file1.readlines()

        for line in Lines:
            videoid = line[:11]
            segmentnumber = line.split()[0].split('_')[-1]
            time = [line.split()[2], line.split()[3]]
            data = [videoid, segmentnumber, time]
            writer.writerow(data)

    cols = ['videoid', 'segmentid']
    df1 = pd.read_csv(preprocess_path)
    df2 = pd.read_csv(withtime_path)
    dfs = [df1, df2]
    df_merged = pd.concat([x.set_index(cols) for x in dfs], axis=1)    
    merged_path = f'{data_dir}merged.csv'
    df_merged.to_csv(merged_path, sep=',', encoding='utf-8')


    lines = []
    with open(merged_path, 'r') as csvfile:
        datareader = csv.DictReader(csvfile)
        for row in datareader:
            video_name = row["videoid"]
            seg_id = row["segmentid"]
            video_path = f"{data_dir}vids/{video_name}.mp4"
            if not os.path.exists(video_path):
                continue

            text = ast.literal_eval(row["text"])
            line = {"text": " ".join(text)}

            vidcap = cv2.VideoCapture(video_path)
            timestamp = ast.literal_eval(row["timestamp"])
            vidcap.set(0,(float(timestamp[1])-float(timestamp[0]))/2);
            ret, frame = vidcap.read() 

            array = np.array(frame, dtype=np.uint8)
            image = Image.fromarray(array, "RGB")
            image_path = f"{data_dir}image/{video_name}_{seg_id}.png"
            image.save(image_path)
            line['image'] = image_path

            with moviepy.editor.VideoFileClip(video_path) as video:
                audio = video.audio
                clip = audio.subclip(timestamp[0], timestamp[1])
                audio_path = f"{data_dir}audio/{video_name}_{seg_id}.wav"
                clip.write_audiofile(audio_path)
            line["audio"] = audio_path

            lines.append(line)

    with open(f"{data_dir}{split}.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, ["audio", "image", "text"])
        writer.writeheader()
        for line in lines:
            writer.writerow(line)
