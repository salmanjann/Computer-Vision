import sys
import numpy as np
from osgeo import gdal
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QLabel, QVBoxLayout, \
    QWidget, QLineEdit, QPushButton, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QPen, QColor, QCursor, QBrush
from PyQt5.QtCore import Qt, QPointF, QRectF

class GeoTiffViewer(QGraphicsView):
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.dataset = None
        self.geo_transform = None
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.setRenderHint(QPainter.Antialiasing)
        self.setMouseTracking(True)  # Enable tracking mouse movements
        self.cursor_label = QLabel(self)  # Floating label for coordinates
        self.cursor_label.setFont(QFont("Arial", 12))
        self.cursor_label.setStyleSheet("background: white; padding: 2px; border: 1px solid black;")
        self.cursor_label.hide()

        self.load_image()
        self.markers = []  # Store marked locations

        # Zoom & Pan Variables
        self._zoom_factor = 1.2
        self._current_scale = 1.0
        self._panning = False
        self._pan_start = QPointF()

        self.setDragMode(QGraphicsView.NoDrag)  # Disable default drag

    def load_image(self):
        """ Load the GeoTIFF image and display it with original size. """
        self.dataset = gdal.Open(self.image_path)
        if self.dataset is None:
            print(f"Failed to open {self.image_path}")
            sys.exit(1)

        self.geo_transform = self.dataset.GetGeoTransform()  # Get geospatial transformation

        # Read RGB or Grayscale data
        if self.dataset.RasterCount >= 3:
            r, g, b = [self.dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(3)]
            image_array = np.dstack((r, g, b)).astype(np.uint8)
            format_type = QImage.Format_RGB888
        else:
            gray_band = self.dataset.GetRasterBand(1).ReadAsArray()
            image_min, image_max = gray_band.min(), gray_band.max()
            if image_max > image_min:
                gray_band = ((gray_band - image_min) / (image_max - image_min) * 255).astype(np.uint8)
            image_array = np.stack((gray_band,) * 3, axis=-1)  # Convert grayscale to RGB
            format_type = QImage.Format_RGB888

        height, width, _ = image_array.shape
        bytes_per_line = width * 3
        q_image = QImage(image_array.data, width, height, bytes_per_line, format_type)

        # Display image in QGraphicsView
        self.pixmap = QPixmap.fromImage(q_image)
        self.image_item = QGraphicsPixmapItem(self.pixmap)
        self.scene.addItem(self.image_item)
        self.setSceneRect(QRectF(self.pixmap.rect()))

    def wheelEvent(self, event):
        """ Zoom in/out at the cursor position. """
        zoom_in = event.angleDelta().y() > 0
        factor = self._zoom_factor if zoom_in else 1 / self._zoom_factor

        self._current_scale *= factor
        if self._current_scale < 0.2:
            self._current_scale = 0.2  # Prevent excessive zoom out
        elif self._current_scale > 10:
            self._current_scale = 10  # Prevent excessive zoom in

        # Zoom at cursor location
        cursor_pos = self.mapToScene(event.pos())
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.scale(factor, factor)
        new_cursor_pos = self.mapToScene(event.pos())
        delta = new_cursor_pos - cursor_pos
        self.translate(delta.x(), delta.y())

        # Change cursor icon
        self.setCursor(QCursor(Qt.CrossCursor))

    def mousePressEvent(self, event):
        """ Start panning when left mouse button is pressed or mark location on right-click. """
        if event.button() == Qt.LeftButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
        elif event.button() == Qt.RightButton:
            self.mark_location(event.pos())
        super().mousePressEvent(event)

    def mark_location(self, pos):
        """ Mark a location with a red dot at the clicked position. """
        scene_pos = self.mapToScene(pos)
        if self.image_item.contains(scene_pos):
            marker = self.scene.addEllipse(scene_pos.x() - 2, scene_pos.y() - 2, 4, 4, QPen(Qt.red), QBrush(Qt.red))
            self.markers.append(marker)

    def mouseMoveEvent(self, event):
        """ Track mouse movement inside the image and allow panning. """
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.translate(delta.x(), delta.y())
        else:
            pos = self.mapToScene(event.pos())  # Convert to scene coordinates
            if self.image_item.contains(pos):
                pixel_x, pixel_y = int(pos.x()), int(pos.y())
                lat, lon = self.pixel_to_geo(pixel_x, pixel_y)
                self.cursor_label.setText(f"Pixel: ({pixel_x}, {pixel_y})\nLat: {lat:.6f}, Lon: {lon:.6f}")
                self.cursor_label.move(event.x() + 15, event.y() - 25)
                self.cursor_label.show()
            else:
                self.cursor_label.hide()

    def mouseReleaseEvent(self, event):
        """ Stop panning when mouse button is released. """
        if event.button() == Qt.LeftButton:
            self._panning = False
            self.setCursor(QCursor(Qt.ArrowCursor))
        super().mouseReleaseEvent(event)

    def pixel_to_geo(self, pixel_x, pixel_y):
        """ Convert pixel (X, Y) to geospatial (Lat, Lon) coordinates. """
        if not self.geo_transform:
            return 0, 0
        origin_x, pixel_width, _, origin_y, _, pixel_height = self.geo_transform
        lon = origin_x + pixel_x * pixel_width
        lat = origin_y + pixel_y * pixel_height
        return lat, lon

    def geo_to_pixel(self, lat, lon):
        """ Convert geospatial (Lat, Lon) to pixel (X, Y) coordinates. """
        if not self.geo_transform:
            return None
        origin_x, pixel_width, _, origin_y, _, pixel_height = self.geo_transform
        pixel_x = int((lon - origin_x) / pixel_width)
        pixel_y = int((lat - origin_y) / pixel_height)
        if 0 <= pixel_x < self.dataset.RasterXSize and 0 <= pixel_y < self.dataset.RasterYSize:
            return pixel_x, pixel_y
        return None

    def mark_geo_location(self, lat, lon):
        """ Mark a location using latitude and longitude. """
        pixel_coords = self.geo_to_pixel(lat, lon)
        if pixel_coords:
            pixel_x, pixel_y = pixel_coords
            marker = self.scene.addEllipse(pixel_x - 2, pixel_y - 2, 4, 4, QPen(Qt.blue), QBrush(Qt.blue))
            self.markers.append(marker)
        else:
            print("Coordinates out of bounds!")


class GeoTiffApp(QWidget):
    def __init__(self, image_path):
        super().__init__()
        self.viewer = GeoTiffViewer(image_path)

        self.lat_input = QLineEdit(self)
        self.lon_input = QLineEdit(self)
        self.lat_input.setPlaceholderText("Enter Latitude")
        self.lon_input.setPlaceholderText("Enter Longitude")

        self.mark_button = QPushButton("Mark Location")
        self.mark_button.clicked.connect(self.mark_location)

        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.clicked.connect(lambda: self.viewer.scale(1.25, 1.25))

        self.zoom_out_button = QPushButton("-")
        self.zoom_out_button.clicked.connect(lambda: self.viewer.scale(0.8, 0.8))

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.lat_input)
        input_layout.addWidget(self.lon_input)
        input_layout.addWidget(self.mark_button)
        input_layout.addWidget(self.zoom_in_button)
        input_layout.addWidget(self.zoom_out_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.viewer)
        main_layout.addLayout(input_layout)

        self.setLayout(main_layout)
        self.setWindowTitle("GeoTIFF Viewer with Marking, Zoom & Pan")

    def mark_location(self):
        try:
            lat = float(self.lat_input.text())
            lon = float(self.lon_input.text())
            self.viewer.mark_geo_location(lat, lon)
        except ValueError:
            print("Invalid input! Please enter valid numeric latitude and longitude.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GeoTiffApp("testGeoTiff.tif")
    window.show()
    sys.exit(app.exec_())
