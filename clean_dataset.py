import os
import shutil
import argparse

import cv2

from clean_utils import preprocess_image_change_detection, compare_frames_change_detection


def are_similar(img1: cv2.Mat, img2: cv2.Mat) -> bool:
    gray = preprocess_image_change_detection(img1, gaussian_blur_radius_list=[9], black_mask=(0,0,0,0))
    gray2 = preprocess_image_change_detection(img2,  gaussian_blur_radius_list=[9], black_mask=(0,0,0,0))
    score, _, __ = compare_frames_change_detection(gray, gray2, 50)
    
    return score <= 1000


def is_valid_img(img: cv2.Mat) -> bool:
    return img is not None and max(img.shape) > 64


def clean(dataset_path, results_path):
    dataset = sorted(os.listdir(dataset_path))
    current_camera = "-1"

    for frame in dataset:
        if not frame.startswith('c'):
            continue

        if not frame.startswith(current_camera):
            current_camera = frame[0:3]
            accepted_imgs = []

        img = cv2.imread(os.path.join(dataset_path, frame))

        if not is_valid_img(img):
            continue

        img = cv2.resize(img, (640,480))

        should_accept_frame = True
        for previous_img in reversed(accepted_imgs):
            if are_similar(previous_img, img):
                should_accept_frame = False
                break

        if should_accept_frame:
            accepted_imgs.append(img)
            shutil.copyfile(os.path.join(dataset_path, frame), os.path.join(results_path, frame))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="./dataset")
    parser.add_argument("--results_path", default="./results")

    args = parser.parse_args()

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    clean(args.dataset_path, args.results_path)