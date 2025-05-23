# GeoTIFF Image Processing and GUI Interaction

GeoTIFF images encode raster data with geospatial metadata, enabling precise linking of each pixel to specific latitude and longitude coordinates. These images are widely used in GIS (Geographic Information System) applications for spatial analysis and visualization.

You are provided with a sample GeoTIFF file named **testGeoTiff.tif**. Your task is to perform the following steps using the **GDAL** library and a GUI framework like **PyQt** or **Tkinter**.

---

## Task-A: Load and Display the Satellite Image

- Load the GeoTIFF file using the GDAL library.
- Display the satellite image in a window.

---

## Task-B: Interactive Mouse Tracking

- Create a GUI that displays the image.
- As the user moves the mouse cursor over the image:
  - Show the pixel coordinates (X, Y) of the cursor in two text boxes.
  - Display the corresponding latitude and longitude of the pixel in two additional text boxes.
  - Update this information in real time.

---

## Task-C: Marking a Specific Location

- Add two input text boxes where users can enter latitude and longitude values.
- Include a **"Mark"** button in the GUI:
  - When clicked, the button should plot a cross at the specified location on the displayed image based on the entered latitude and longitude.

---

## Task-D (Bonus): Zoom and Pan Functionality

Enhance the GUI with:

- **Zooming functionality** using the mouse wheel.
- **Panning functionality** via mouse drag.

---