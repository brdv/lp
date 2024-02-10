from engine import detect, process, recognise, detect_belg, post_process
import cv2
import numpy
import argparse
import os
import glob
import sys
import time

from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--i", "-image", help="Input image path", type=str)
parser.add_argument("--v", "-video", help="Input video path", type=str)

args = parser.parse_args()
abs_path = os.path.dirname(sys.executable)


def get_time_string() -> str:
    return datetime.now().strftime("%Y%m%d%H:%M")


def img_to_lp(file: str):
    file_name = file.split("/")[-1]
    folder = f"reads/{get_time_string()}/{file_name}"
    lps = []

    try:
        os.makedirs(folder)
    except:
        files = glob.glob("tmp")
        for f in files:
            os.remove(f)

    input_image = cv2.imread(file)
    # Als je in een eerder stadium 

    detection, crops = detect(input_image)

    i = 1
    for crop in crops:
        crop = process(crop)

        cv2.imwrite(f"{folder}/crop" + str(i) + ".jpg", crop)
        recognise(f"{folder}/crop" + str(i) + ".jpg", f"{folder}/crop" + str(i))
        lp = post_process(f"{folder}/crop" + str(i) + ".txt")
        # todo: add copy of original image.
        lp = lp.rstrip()
        lps.append(lp)
        i += 1

    cv2.imwrite(f"{folder}/detection.jpg", detection)

    return lps


folder_path = "/Users/brdv/repos/personal/license_processor/src/imgs"

results: list[bool] = []

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    lps = img_to_lp(file_path)
    lp = "---"

    if "polo" in file_path:
        lp = "JL734Z"

    if "toer" in file_path:
        lp = "L655RT"

    correct = lp in lps
    results.append(correct)
    print(f"for file: {filename} the ANRP result is: {correct}")

total = len(results)
t_count = results.count(True)
f_count = results.count(False)

print(
    f"\nTotal items is {total}, of which {t_count} are successful and {f_count} have failed."
)

print(f"This is a success-rate of {t_count/total}")
