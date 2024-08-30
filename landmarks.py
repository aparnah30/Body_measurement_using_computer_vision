import logging
import matplotlib.pyplot as plt
import os
import warnings
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tabulate import tabulate
import math
import argparse
import cv2
from mediapipe.python.solutions import (
    pose,
)
import yaml
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="google.protobuf",
)

LANDMARK_NAME_TO_INDEX = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

class Landmarker:

    resized_height = 256
    resized_width = 256

    def __init__(self) -> None:
        self.args = self.parse_args()
        self.measurements = self.load_landmarks()
        if self.args.front_image is None:
            raise Exception("front image needs to be passed")
        if self.args.side_image is None:
            raise Exception("side image needs to be passed")

        self.front_image = cv2.imread(self.args.front_image)
        self.side_image = cv2.imread(self.args.side_image)

        self.front_image_resized = cv2.resize(self.front_image, (self.resized_height, self.resized_width))
        self.side_image_resized = cv2.resize(self.side_image, (self.resized_height, self.resized_width))

        self.distances = {}

        self.person_height = self.args.person_height
        self.pixel_height = self.args.pixel_height

        self.pose = pose.Pose(
            static_image_mode=True,
            min_detection_confidence=self.args.pose_detection_confidence,
            min_tracking_confidence=self.args.pose_tracking_confidence,
        )

        self.landmarks_indices = [
            LANDMARK_NAME_TO_INDEX["left_shoulder"],
            LANDMARK_NAME_TO_INDEX["right_shoulder"],
            LANDMARK_NAME_TO_INDEX["left_elbow"],
            LANDMARK_NAME_TO_INDEX["right_elbow"],
            LANDMARK_NAME_TO_INDEX["left_wrist"],
            LANDMARK_NAME_TO_INDEX["right_wrist"],
            LANDMARK_NAME_TO_INDEX["left_hip"],
            LANDMARK_NAME_TO_INDEX["right_hip"],
            LANDMARK_NAME_TO_INDEX["left_knee"],
            LANDMARK_NAME_TO_INDEX["right_knee"],
            LANDMARK_NAME_TO_INDEX["left_ankle"],
            LANDMARK_NAME_TO_INDEX["right_ankle"],
        ]

    def load_landmarks(self):
        with open(self.args.yaml_file, "r") as file:
            landmarks_data = yaml.safe_load(file)
        measurements = {}
        for measurement in landmarks_data["measurements"]:
            measurements[measurement["name"]] = [LANDMARK_NAME_TO_INDEX[l] for l in measurement["landmarks"]]
        return measurements

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--front",
            dest="front_image",
            type=str,
            help="Front image",
        )
        parser.add_argument(
            "--side",
            dest="side_image",
            type=str,
            help="Side image",
        )
        parser.add_argument(
            "--pose_detection_confidence",
            dest="pose_detection_confidence",
            default=0.5,
            type=float,
            help="Confidence score for pose detection",
        )
        parser.add_argument(
            "--pose_tracking_confidence",
            dest="pose_tracking_confidence",
            default=0.5,
            type=float,
            help="Confidence score for pose tracking",
        )
        parser.add_argument(
            "--person_height",
            dest="person_height",
            type=int,
            help="person height of person",
        )
        parser.add_argument(
            "--pixel_height",
            dest="pixel_height",
            type=int,
            help="pixel height of person",
        )
        parser.add_argument(
            "--measurement",
            dest="measurement",
            nargs="+",
            type=str,
            help="Type of measurement",
        )
        parser.add_argument(
            "--yaml_file",
            dest="yaml_file",
            type=str,
            help="Path to the YAML file containing landmarks",
        )
        return parser.parse_args()
    
    def run(self):
        front_results, _ = self.process_images()

        self.get_center_top_point(front_results)

        table = []
        if self.args.measurement:
            for m in self.args.measurement:
                if m not in self.measurements:
                    raise Exception("Incorrect input (input not present in config.yml)")
                else:
                    distance = self.calculate_distance_betn_landmarks(front_results, m)
                    table.append([m, distance])
        else:
            for m in self.measurements:
                distance = self.calculate_distance_betn_landmarks(front_results, m)
                table.append([m, distance])
        

        output = tabulate(
            table,
            headers=[
                "measurement",
                "Distance (cm)",
            ],
            tablefmt="plain",
        )
        print(output)

        self.pose.close()

    def process_images(self):
        front_results = self.pose.process(
            cv2.cvtColor(
                self.front_image_resized,
                cv2.COLOR_BGR2RGB,
            )
        )
        side_results = self.pose.process(
            cv2.cvtColor(
                self.side_image_resized,
                cv2.COLOR_BGR2RGB,
            )
        )

        self.side_image_keypoints = self.side_image_resized.copy()
        self.front_image_keypoints = self.front_image_resized.copy()

        if front_results.pose_landmarks:  # type: ignore
            self.draw_landmarks(
                self.front_image_keypoints,
                front_results.pose_landmarks,  # type: ignore
                self.landmarks_indices,
            )
        if side_results.pose_landmarks:  # type: ignore
            self.draw_landmarks(
                self.side_image_keypoints,
                side_results.pose_landmarks,  # type: ignore# type: ignore
                self.landmarks_indices,
            )

# Save images to disk
        cv2.imwrite('side_image_keypoints.png', self.side_image_keypoints)
        cv2.imwrite('front_image_keypoints.png', self.front_image_keypoints)

        return (
            front_results,
            side_results,
        )

    def pixel_to_metric_ratio(self):
        self.pixel_height = self.pixel_distance * 2
        pixel_to_metric_ratio = self.person_height / self.pixel_height
        logging.debug(
            "pixel_to_metric_ratio %s",
            pixel_to_metric_ratio,
        )
        return pixel_to_metric_ratio

    def draw_landmarks(self, image, landmarks, indices):
        for idx in indices:
            landmark = landmarks.landmark[idx]
            h, w, _ = image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            self.circle(image, cx, cy)

    def circle(self, image, cx, cy):
        return cv2.circle(
            image,
            (cx, cy),
            2,
            (255, 0, 0),
            -1,
        )

    def calculate_distance_betn_landmarks(
        self,
        front_results,
        measurement_name,
    ):
        if not front_results.pose_landmarks:
            return

        landmarks = front_results.pose_landmarks.landmark
        landmark_names = self.measurements[measurement_name]

        total_distance = 0
        for idx in range(len(landmark_names) - 1):
            _current = landmarks[landmark_names[idx]]
            _next = landmarks[landmark_names[idx + 1]]
            pixel_distance = self.euclidean_distance(
                _current.x * self.resized_width,
                _current.y * self.resized_height,
                _next.x * self.resized_width,
                _next.y * self.resized_height,
            )
            real_distance = pixel_distance * self.pixel_to_metric_ratio()
            total_distance += real_distance
        return total_distance

    def euclidean_distance(self, x1, y1, x2, y2):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance


    def get_center_top_point(self, side_results):
        gray_image = cv2.cvtColor(
            self.side_image_keypoints,
            cv2.COLOR_BGR2GRAY,
        )
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        roi = blurred_image[
            0 : int(self.side_image_resized.shape[0] / 2),
            :,
        ]
        edges = cv2.Canny(roi, 50, 150)
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        xt, yt = None, None
        topmost_point = None
    
        if contours:
            largest_contour = max(
                contours,
                key=cv2.contourArea,
            )
            topmost_point = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
            xt, yt = topmost_point
    
            cv2.circle(
                self.side_image_keypoints,
                (xt, yt),
                2,
                (255, 255, 0),
                -1,
            )
    
        xc, yc = None, None
        landmarks = side_results.pose_landmarks.landmark
    
        if side_results.pose_landmarks:
            left_hip = landmarks[LANDMARK_NAME_TO_INDEX["left_hip"]]
            right_hip = landmarks[LANDMARK_NAME_TO_INDEX["right_hip"]]
            center_point = (
                (left_hip.x + right_hip.x) / 2,
                (left_hip.y + right_hip.y) / 2,
            )
            center_point = (
                int(center_point[0] * self.side_image_resized.shape[1]),
                int(center_point[1] * self.side_image_resized.shape[0]),
            )
            xc, yc = center_point
            self.circle(
                self.side_image_keypoints,
                xc,
                yc,
            )
    
            self.pixel_distance = self.euclidean_distance(xc, yc, xt, yt)
            logging.debug(
                "top_center_pixel_distance: %s",
                self.pixel_distance,
            )
            self.pixel_height = self.pixel_distance * 2
            logging.debug(
                "pixel height: %s ",
                self.pixel_height,
            )
            self.distance = self.euclidean_distance(xc, yc, xt, yt) * self.pixel_to_metric_ratio()
    
        return self.distance

if __name__ == "__main__":
    landmarker = Landmarker()
    landmarker.run()
