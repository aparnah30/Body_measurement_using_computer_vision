# Body Measurement Using Computer Vision

This project utilizes computer vision techniques to estimate body measurements from front and side images of a person. It uses OpenCV and Mediapipe for landmark detection.

## Setup

### Requirements

- Docker
- Mediapipe

### Install Docker

Follow the instructions on the [official Docker website](https://docs.docker.com/get-docker/) to install Docker on your system.

## How to Run the Code

### Using Docker

1. **Build the Docker image:**

    ```sh
    docker build -t <docker_image> .
    ```

2. **Run the Docker container:**

    ```sh
    docker run -it <docker_image> --front <path/to/front_image.jpg> --side <path/to/side_image.jpg> --person_height <height_in_cm> --yaml_file <pth/to/config.yml>
    ```

### Command Line Arguments

- `--front`: Path to the front image.
- `--side`: Path to the side image.
- `--pose_detection_confidence`: (Optional) Confidence score for pose detection (default: 0.5).
- `--pose_tracking_confidence`: (Optional) Confidence score for pose tracking (default: 0.5).
- `--person_height`: Height of the person in centimeters.
- `--pixel_height`: (Optional) Pixel height of the person.
- `--measurement`: (Optional) Type of measurement.
- `--yaml_file`: Path to the YAML file containing landmarks.

### Example Command

```sh
docker run -it landmarks --front ./assets/aparna_front.jpg --side ./assets/aparna_side.jpg --person_height 157 --yaml_file config.yml
```
