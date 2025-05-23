import sys
from osgeo import gdal
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint
import pickle


class ImageLoader(QWidget):
    def __init__(self):
        super().__init__()
        self.sift = cv2.SIFT_create()
        self.refKeypoints, self.refDescriptors = None, None
        self.homographyMatrix = None
        self.geoTransform = None  # For coordinate verification
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Automatic Georeferencing System")
        self.setGeometry(100, 100, 800, 600)

        self.aerialDropdown = QComboBox(self)
        self.aerialDropdown.addItems([f"dataset/aerial{i}.png" for i in range(1, 15)])
        self.aerialDropdown.currentIndexChanged.connect(self.loadAerialImage)

        self.refLabel = QLabel("Reference Image will appear here")
        self.refLabel.setFixedSize(800, 250)
        self.aerialLabel = QLabel("Aerial Image will appear here")
        self.aerialLabel.setMouseTracking(True)
        self.aerialLabel.mousePressEvent = self.getPixelCoordinates

        layout = QVBoxLayout()
        layout.addWidget(self.refLabel)
        layout.addWidget(self.aerialDropdown)
        layout.addWidget(self.aerialLabel)

        self.setLayout(layout)

        # Load default images and precompute reference keypoints
        self.loadReferenceImage()
        self.loadAerialImage()

    def loadReferenceImage(self):
        filePath = "dataset/reference1Km.tif"
        dataset = gdal.Open(filePath)
        if dataset:
            self.geoTransform = dataset.GetGeoTransform()  # Store geotransform for lat/lon conversion
            img = dataset.ReadAsArray()
            img = np.array(img, dtype=np.uint8)
            if len(img.shape) == 3:
                img = np.transpose(img, (1, 2, 0))
            self.displayImage(img, self.refLabel)
            self.refKeypoints, self.refDescriptors = self.computeKeypointsAndDescriptors(img)
            serializableKeypoints = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in
                                     self.refKeypoints]
            with open("dataset/reference_features.pkl", "wb") as f:
                pickle.dump((serializableKeypoints, self.refDescriptors), f)

    def loadAerialImage(self):
        filePath = self.aerialDropdown.currentText()
        if filePath:
            img = cv2.imread(filePath)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.displayImage(img, self.aerialLabel)
                keypoints, descriptors = self.computeKeypointsAndDescriptors(img)
                print(f"Extracted {len(keypoints)} keypoints from aerial image.")
                self.homographyMatrix = self.estimateHomography(keypoints, descriptors)

    def displayImage(self, img, label):
        height, width = img.shape[:2]
        channel = 3 if len(img.shape) == 3 else 1
        bytesPerLine = channel * width
        qImg = QImage(img.tobytes(), width, height, bytesPerLine,
                      QImage.Format_RGB888 if channel == 3 else QImage.Format_Grayscale8)
        label.setPixmap(QPixmap.fromImage(qImg))
        label.setScaledContents(True)

    def getPixelCoordinates(self, event):
        if event.button() == Qt.LeftButton:
            x = event.pos().x()
            y = event.pos().y()
            print(f"Clicked Coordinates: ({x}, {y})")
            if self.homographyMatrix is not None:
                lat, lon = self.projectToReferenceImage(x, y)
                print(f"Projected Location: Latitude {lat}, Longitude {lon}")
                self.drawCoordinatesOnImage(x, y, lat, lon)

    def projectToReferenceImage(self, x, y):
        point = np.array([[x, y, 1]], dtype=np.float32).T
        projected = np.dot(self.homographyMatrix, point)
        projected /= projected[2]  # Normalize
        if self.geoTransform:
            lon = self.geoTransform[0] + projected[0, 0] * self.geoTransform[1] + projected[1, 0] * self.geoTransform[2]
            lat = self.geoTransform[3] + projected[0, 0] * self.geoTransform[4] + projected[1, 0] * self.geoTransform[5]
            return lat, lon
        return projected[0, 0], projected[1, 0]

    def drawCoordinatesOnImage(self, x, y, lat, lon):
        img = cv2.imread(self.aerialDropdown.currentText())
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.putText(img, f"({lat:.5f}, {lon:.5f})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            self.displayImage(img, self.aerialLabel)

    def computeKeypointsAndDescriptors(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def estimateHomography(self, keypoints, descriptors):
        if self.refKeypoints is None or self.refDescriptors is None:
            print("Reference descriptors not loaded!")
            return None

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors, self.refDescriptors, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) >= 4:  # Reduced threshold
            src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.refKeypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            print("Homography Matrix:")
            print(H)
            return H
        else:
            print(f"Not enough matches found: {len(good_matches)}/4")
            return None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageLoader()
    window.show()
    sys.exit(app.exec_())
