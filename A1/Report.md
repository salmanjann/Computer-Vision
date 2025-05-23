# GeoTIFF Viewer: GDAL Functionality Overview

## Introduction
This project is a GeoTIFF viewer built with PyQt5 and GDAL, allowing users to visualize and interact with geospatial images. The application supports zooming, panning, coordinate display, and location marking using pixel or geospatial coordinates.

## GDAL Functionalities Used

### 1. Opening a GeoTIFF File
```python
self.dataset = gdal.Open(self.image_path)
```
- Uses `gdal.Open()` to read the GeoTIFF file.
- If the file fails to open, the program exits.

### 2. Extracting Raster Bands
```python
r, g, b = [self.dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(3)]
```
- Reads individual raster bands (Red, Green, Blue) from the dataset.
- If the image has only one band (grayscale), it normalizes the pixel values to a `0-255` range.

### 3. Converting Image Data to QImage
```python
image_array = np.dstack((r, g, b)).astype(np.uint8)
q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
```
- Stacks the extracted bands into an RGB image.
- Converts the NumPy array into a `QImage` for display in PyQt.

### 4. Geospatial Transformation
#### **Extracting GeoTransform Data**
```python
self.geo_transform = self.dataset.GetGeoTransform()
```
- Retrieves the affine transformation parameters, mapping pixel coordinates to geographic coordinates.
- `geo_transform` contains:
  - `origin_x, pixel_width, rotation_x, origin_y, rotation_y, pixel_height`

#### **Pixel to Geographic Coordinates Conversion**
```python
def pixel_to_geo(self, pixel_x, pixel_y):
    origin_x, pixel_width, _, origin_y, _, pixel_height = self.geo_transform
    lon = origin_x + pixel_x * pixel_width
    lat = origin_y + pixel_y * pixel_height
    return lat, lon
```
- Converts pixel `(x, y)` positions to latitude and longitude using the `geo_transform` matrix.

#### **Geographic to Pixel Coordinates Conversion**
```python
def geo_to_pixel(self, lat, lon):
    origin_x, pixel_width, _, origin_y, _, pixel_height = self.geo_transform
    pixel_x = int((lon - origin_x) / pixel_width)
    pixel_y = int((lat - origin_y) / pixel_height)
    return pixel_x, pixel_y
```
- Converts latitude and longitude back to pixel coordinates.
- Ensures the pixel coordinates are within the image bounds.

## Conclusion
This application utilizes GDAL to extract raster data, perform geospatial transformations, and integrate with PyQt for interactive visualization. The ability to map between pixel and geospatial coordinates allows for precise location marking and data interpretation.
