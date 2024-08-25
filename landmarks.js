const fs = require("fs");
const path = require("path");
const { ArgumentParser } = require("argparse");
const cv = require("@techstark/opencv-js");
const yaml = require("js-yaml");
const { Pose, POSE_LANDMARKS } = require("@mediapipe/pose");

const logging = console;
const warnings = console;

class Landmarker {
  static resizedHeight = 256;
  static resizedWidth = 256;

  constructor() {
    this.args = this.parseArgs();
    this.measurements = this.loadLandmarks();
    if (!this.args.frontImage) {
      throw new Error("Front image needs to be passed");
    }
    if (!this.args.sideImage) {
      throw new Error("Side image needs to be passed");
    }

    this.frontImage = cv.imread(this.args.frontImage);
    this.sideImage = cv.imread(this.args.sideImage);

    this.frontImageResized = cv.resize(
      this.frontImage,
      new cv.Size(Landmarker.resizedWidth, Landmarker.resizedHeight),
    );
    this.sideImageResized = cv.resize(
      this.sideImage,
      new cv.Size(Landmarker.resizedWidth, Landmarker.resizedHeight),
    );

    this.distances = {};

    this.personHeight = this.args.personHeight;
    this.pixelHeight = this.args.pixelHeight;

    this.pose = new Pose({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
      },
    });

    this.landmarksIndices = [
      POSE_LANDMARKS.LEFT_SHOULDER,
      POSE_LANDMARKS.RIGHT_SHOULDER,
      POSE_LANDMARKS.LEFT_ELBOW,
      POSE_LANDMARKS.RIGHT_ELBOW,
      POSE_LANDMARKS.LEFT_WRIST,
      POSE_LANDMARKS.RIGHT_WRIST,
      POSE_LANDMARKS.LEFT_HIP,
      POSE_LANDMARKS.RIGHT_HIP,
      POSE_LANDMARKS.LEFT_KNEE,
      POSE_LANDMARKS.RIGHT_KNEE,
      POSE_LANDMARKS.LEFT_ANKLE,
      POSE_LANDMARKS.RIGHT_ANKLE,
    ];
  }

  loadLandmarks() {
    const file = fs.readFileSync(this.args.yamlFile, "utf8");
    const landmarksData = yaml.load(file);
    const measurements = {};
    for (const measurement of landmarksData.measurements) {
      measurements[measurement.name] = measurement.landmarks;
    }
    return measurements;
  }

  parseArgs() {
    const parser = new ArgumentParser({
      description: "Process images and calculate measurements",
    });
    parser.add_argument("--front", {
      dest: "frontImage",
      required: true,
      help: "Path to the front image",
    });
    parser.add_argument("--side", {
      dest: "sideImage",
      required: true,
      help: "Path to the side image",
    });
    parser.add_argument("--poseDetectionConfidence", {
      dest: "poseDetectionConfidence",
      default: 0.5,
      type: "float",
      help: "Confidence score for pose detection",
    });
    parser.add_argument("--poseTrackingConfidence", {
      dest: "poseTrackingConfidence",
      default: 0.5,
      type: "float",
      help: "Confidence score for pose tracking",
    });
    parser.add_argument("--personHeight", {
      dest: "personHeight",
      required: true,
      type: "int",
      help: "Person height in cm",
    });
    parser.add_argument("--pixelHeight", {
      dest: "pixelHeight",
      type: "int",
      help: "Pixel height of person",
    });
    parser.add_argument("--measurement", {
      dest: "measurement",
      nargs: "+",
      type: "str",
      help: "Type of measurement",
    });
    parser.add_argument("--yamlFile", {
      dest: "yamlFile",
      required: true,
      help: "Path to the YAML file containing landmarks",
    });
    return parser.parse_args();
  }

  async run() {
    await this.pose.initialize();
    const { frontResults, sideResults } = await this.processImages();

    this.getCenterTopPoint(sideResults);

    const table = [];
    if (this.args.measurement) {
      for (const m of this.args.measurement) {
        if (!this.measurements[m]) {
          throw new Error("Incorrect input (input not present in config.yml)");
        } else {
          const distance = this.calculateDistanceBetweenLandmarks(
            frontResults,
            m,
          );
          table.push([m, distance]);
        }
      }
    } else {
      for (const m in this.measurements) {
        const distance = this.calculateDistanceBetweenLandmarks(
          frontResults,
          m,
        );
        table.push([m, distance]);
      }
    }

    console.table(table);

    this.pose.close();
  }

  async processImages() {
    const frontResults = await this.pose.estimatePoses(this.frontImageResized);
    const sideResults = await this.pose.estimatePoses(this.sideImageResized);

    this.sideImageKeypoints = this.sideImageResized.clone();
    this.frontImageKeypoints = this.frontImageResized.clone();

    if (frontResults[0].landmarks) {
      this.drawLandmarks(
        this.frontImageKeypoints,
        frontResults[0].landmarks,
        this.landmarksIndices,
      );
    }
    if (sideResults[0].landmarks) {
      this.drawLandmarks(
        this.sideImageKeypoints,
        sideResults[0].landmarks,
        this.landmarksIndices,
      );
    }
    return {
      frontResults: frontResults[0],
      sideResults: sideResults[0],
    };
  }

  pixelToMetricRatio() {
    const pixelToMetricRatio = this.personHeight / this.pixelHeight;
    logging.debug("pixelToMetricRatio %s", pixelToMetricRatio);
    return pixelToMetricRatio;
  }

  drawLandmarks(image, landmarks, indices) {
    for (const idx of indices) {
      const landmark = landmarks[idx];
      const h = image.rows;
      const w = image.cols;
      const cx = Math.round(landmark.x * w);
      const cy = Math.round(landmark.y * h);
      this.circle(image, cx, cy);
    }
  }

  circle(image, cx, cy) {
    cv.circle(image, new cv.Point(cx, cy), 2, new cv.Scalar(255, 0, 0), -1);
  }

  calculateDistanceBetweenLandmarks(frontResults, measurementName) {
    if (!frontResults.landmarks) {
      return;
    }

    const landmarks = frontResults.landmarks;
    const landmarkNames = this.measurements[measurementName];

    let totalDistance = 0;
    for (let i = 0; i < landmarkNames.length - 1; i++) {
      const current = landmarks[landmarkNames[i]];
      const next = landmarks[landmarkNames[i + 1]];
      const pixelDistance = this.euclideanDistance(
        current.x * Landmarker.resizedWidth,
        current.y * Landmarker.resizedHeight,
        next.x * Landmarker.resizedWidth,
        next.y * Landmarker.resizedHeight,
      );
      const realDistance = pixelDistance * this.pixelToMetricRatio();
      totalDistance += realDistance;
    }
    return totalDistance;
  }

  euclideanDistance(x1, y1, x2, y2) {
    return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
  }

  getCenterTopPoint(sideResults) {
    const grayImage = cv.cvtColor(this.sideImageKeypoints, cv.COLOR_BGR2GRAY);
    const blurredImage = cv.GaussianBlur(grayImage, new cv.Size(5, 5), 0);
    const roi = blurredImage.roi(
      new cv.Rect(
        0,
        0,
        this.sideImageResized.cols,
        Math.floor(this.sideImageResized.rows / 2),
      ),
    );
    this.edges = cv.Canny(roi, 50, 150);
    const contours = this.edges.findContours(
      cv.RETR_EXTERNAL,
      cv.CHAIN_APPROX_SIMPLE,
    );

    let xt, yt;
    this.topmostPoint = null;

    for (const contour of contours) {
      const [xt, yt] = contour.minEnclosingCircle();
      if (this.topmostPoint === null || yt < this.topmostPoint[1]) {
        this.topmostPoint = [xt, yt];
      }
    }

    const { x, y } = sideResults.landmarks[POSE_LANDMARKS.NOSE];
    const centerPoint = [
      x * Landmarker.resizedWidth,
      y * Landmarker.resizedHeight,
    ];
    this.pixelHeight = Math.abs(centerPoint[1] - this.topmostPoint[1]);

    cv.circle(
      this.sideImageKeypoints,
      new cv.Point(centerPoint[0], centerPoint[1]),
      2,
      new cv.Scalar(255, 0, 0),
      -1,
    );
    cv.circle(
      this.sideImageKeypoints,
      new cv.Point(this.topmostPoint[0], this.topmostPoint[1]),
      2,
      new cv.Scalar(255, 0, 0),
      -1,
    );
  }
}

const landmarker = new Landmarker();
landmarker.run().catch((error) => {
  console.error(error);
});
