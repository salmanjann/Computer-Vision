# Two Dice Game using Object Detection
# Imports
import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import yaml
from pathlib import Path
import torch
import random
from ultralytics import YOLO
import shutil

# dataset = "https://www.kaggle.com/datasets/nomihsa965/dice-detection-upper-view"

# Configuration
MODEL_PATH = "runs/detect/train3/weights/best.pt"  # Path to trained model
IMAGE_SIZE = (832, 832)  # Size of display image
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detection


class DiceGameApp:
    def __init__(self, root):
        """Initialize the application"""
        self.root = root
        self.root.title("Two Dice Game")
        self.root.geometry("950x950")
        self.root.resizable(False, False)

        # Game state variables
        self.score = 0
        self.game_active = False
        self.current_image = None
        self.cap = None
        self.use_camera = False
        self.test_images = []
        self.current_test_image_idx = 0

        # Load the trained model
        try:
            self.model = YOLO(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.model = None

        # Build the UI
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface"""
        # Frame for controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        # Start/Reset button
        self.start_button = tk.Button(control_frame, text="Start Game", command=self.start_game, width=15, height=2)
        self.start_button.pack(side=tk.LEFT, padx=10)

        # Score button
        self.score_button = tk.Button(control_frame, text="Score Dice", command=self.score_dice, width=15, height=2,
                                      state=tk.DISABLED)
        self.score_button.pack(side=tk.LEFT, padx=10)

        # Exit button
        exit_button = tk.Button(control_frame, text="Exit", command=self.root.destroy, width=15, height=2)
        exit_button.pack(side=tk.LEFT, padx=10)

        # Input selection frame
        input_frame = tk.Frame(self.root)
        input_frame.pack(pady=10)

        # Camera option
        self.camera_button = tk.Button(input_frame, text="Use Camera", command=self.use_camera_input, width=15)
        self.camera_button.pack(side=tk.LEFT, padx=10)

        # Test images option
        self.test_images_button = tk.Button(input_frame, text="Use Test Images", command=self.use_test_images, width=15)
        self.test_images_button.pack(side=tk.LEFT, padx=10)

        # Score display
        score_frame = tk.Frame(self.root)
        score_frame.pack(pady=10)

        tk.Label(score_frame, text="Current Score:", font=("Arial", 16)).pack(side=tk.LEFT, padx=10)
        self.score_var = tk.StringVar(value="0")
        tk.Label(score_frame, textvariable=self.score_var, font=("Arial", 16, "bold")).pack(side=tk.LEFT)

        # Status display
        self.status_var = tk.StringVar(value="Game not started")
        tk.Label(self.root, textvariable=self.status_var, font=("Arial", 12)).pack(pady=5)

        # Image display canvas
        self.canvas = tk.Canvas(self.root, width=IMAGE_SIZE[0], height=IMAGE_SIZE[1], bg="black")
        self.canvas.pack(pady=10)

        # Display instructions
        instructions = "Game Rules:\n" \
                       "1. Click 'Start Game' to begin\n" \
                       "2. Throw two dice\n" \
                       "3. Click 'Score Dice' to capture and score\n" \
                       "4. If both dice show the same value, the game ends\n" \
                       "5. Otherwise, the sum is added to your score and you can throw again"
        tk.Label(self.root, text=instructions, justify=tk.LEFT, font=("Arial", 10)).pack(pady=10)

    def start_game(self):
        """Start or reset the game"""
        # Reset score
        self.score = 0
        self.score_var.set("0")
        self.game_active = True

        # Update UI states
        self.score_button.config(state=tk.NORMAL)
        self.start_button.config(text="Reset Game")
        self.status_var.set("Game started! Throw your dice and click 'Score Dice'")

        # Reset test image index if using test images
        self.current_test_image_idx = 0

        # Update display
        self.update_display()

    def use_camera_input(self):
        """Switch to camera input"""
        if self.cap is not None:
            self.cap.release()

        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")

            self.use_camera = True
            self.test_images = []
            self.status_var.set("Using camera input")
            self.update_display()
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to access camera: {e}")
            self.use_camera = False

    def use_test_images(self):
        """Switch to test images input"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        folder_path = filedialog.askdirectory(title="Select folder with test images")
        if folder_path:
            self.test_images = sorted([
                os.path.join(folder_path, file)
                for file in os.listdir(folder_path)
                if file.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

            # Shuffle the test images as requested
            random.shuffle(self.test_images)

            if not self.test_images:
                messagebox.showwarning("No Images", "No image files found in the selected folder")
                return

            self.use_camera = False
            self.current_test_image_idx = 0
            self.status_var.set(f"Using test images: {len(self.test_images)} images found")
            self.update_display()

    def update_display(self):
        """Update the image display"""
        if self.use_camera and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.current_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.show_image(self.current_image)

                # Continue updating if game is active
                if self.game_active:
                    self.root.after(30, self.update_display)
            else:
                self.status_var.set("Camera error: Cannot read frame")

        elif self.test_images and not self.use_camera:
            if self.current_test_image_idx < len(self.test_images):
                image_path = self.test_images[self.current_test_image_idx]
                image = cv2.imread(image_path)
                if image is not None:
                    self.current_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.show_image(self.current_image)
                    self.status_var.set(f"Test image {self.current_test_image_idx + 1}/{len(self.test_images)}")
                else:
                    self.status_var.set(f"Error loading image {image_path}")
            else:
                self.status_var.set("No more test images available")

    def show_image(self, image):
        """Display an image on the canvas"""
        if image is None:
            print("Warning: Attempted to display None image")
            return
        try:
            # Clear the canvas before drawing new image
            self.canvas.delete("all")
            # Rest of the image display code...
        except Exception as e:
            print(f"Error displaying image: {e}")

        # Resize image to fit canvas
        image = cv2.resize(image, (IMAGE_SIZE[0], IMAGE_SIZE[1]))

        # Convert to PhotoImage
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)

        # Update canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo  # Keep a reference to prevent garbage collection

    def score_dice(self):
        """Process the current image and score the dice"""
        if not self.game_active or self.current_image is None:
            return

        # Use the model to detect dice
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded")
            return

        # Make a copy of the current image for processing
        image_for_detection = self.current_image.copy()

        # Run detection on the image
        results = self.model(image_for_detection, conf=CONFIDENCE_THRESHOLD,imgsz=832)

        # Add this line to print more details about the entire results
        print(f"Results summary: {len(results)} detection(s) performed")

        # Process the results
        dice_values = []
        if results and len(results) > 0:
            result = results[0]  # Get first result

            # Get detected boxes and classes
            boxes = result.boxes

            print(f"Detected {len(boxes)} dice in the image")
            # Draw bounding boxes on the image
            annotated_image = image_for_detection.copy()

            for box in boxes:
                # Extract information
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                # Map class_id to actual dice value (in our case, class 0 = dice face 1, etc.)
                dice_value = class_id + 1
                dice_values.append(dice_value)
                # Print detailed information about each detection
                print(
                    f"Die #: Value={dice_value}, Class ID={class_id}, Confidence={confidence:.4f}, Box=({x1},{y1},{x2},{y2})")

                # Draw rectangle
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label
                label = f"{dice_value}: {confidence:.2f}"
                cv2.putText(annotated_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display annotated image
            self.show_image(annotated_image)

            # If we have 2 dice detected, process the game logic
            if len(dice_values) == 2:
                dice1, dice2 = dice_values
                self.process_dice_values(dice1, dice2)
            else:
                messagebox.showwarning("Detection Issue",
                                       f"Expected 2 dice, but detected {len(dice_values)}. Please try again.")

        else:
            messagebox.showwarning("No Detection", "No dice detected in the image")


        # If using test images, move to the next image
        if not self.use_camera and self.test_images:
            self.current_test_image_idx += 1
            self.update_display()

    def process_dice_values(self, dice1, dice2):
        """Process the dice values and update the game state"""
        # Display the values
        self.status_var.set(f"Dice values: {dice1} and {dice2}")

        # Check if the dice have the same value
        if dice1 == dice2:
            messagebox.showinfo("Game Over", f"Both dice show {dice1}. Game over! Final score: {self.score}")
            self.score_button.config(state=tk.DISABLED)
            self.game_active = False
        else:
            # Add the sum to the score
            self.score += (dice1 + dice2)
            self.score_var.set(str(self.score))

            # Continue the game
            self.status_var.set(f"Added {dice1 + dice2} to your score. Throw again!")


def create_yaml_file():
    """Create a YAML file for training if it doesn't exist"""
    # Get the absolute path to the current directory
    current_dir = os.path.abspath(os.getcwd())

    yaml_content = f"""
# Dataset configuration for dice detection
train: {current_dir}/dataset/train/images  # Train images folder
val: {current_dir}/dataset/val/images      # Validation images folder
test: {current_dir}/dataset/test/images    # Test images folder

# Number of classes
nc: 6  # 6 dice faces (1-6)

# Class names
names:
  0: 1
  1: 2
  2: 3
  3: 4
  4: 5
  5: 6
"""

    with open('dice_dataset.yaml', 'w') as f:
        f.write(yaml_content)

    print("Created dice_dataset.yaml file with absolute paths")


def split_data(data_dir='data', train_pct=0.6, val_pct=0.2, test_pct=0.2):
    """Split the dataset into train, validation, and test sets"""
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')

    # Create output directories if they don't exist
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join('dataset', split, 'images'), exist_ok=True)
        os.makedirs(os.path.join('dataset', split, 'labels'), exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)

    # Calculate split points
    total_files = len(image_files)
    train_end = int(total_files * train_pct)
    val_end = train_end + int(total_files * val_pct)

    # Split the data
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    # Function to copy files
    def copy_files(file_list, split):
        for file in file_list:
            # Copy image
            src_img = os.path.join(images_dir, file)
            dst_img = os.path.join('dataset', split, 'images', file)

            # Copy corresponding label (assuming same name with .txt extension)
            label_file = os.path.splitext(file)[0] + '.txt'
            src_lbl = os.path.join(labels_dir, label_file)
            dst_lbl = os.path.join('dataset', split, 'labels', label_file)

            # Copy files if they exist using shutil (cross-platform)
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)

            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)

    # Copy files to their respective directories
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    print(f"Data split completed: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")


def train_model():
    """Train the YOLOv8 model"""
    try:
        # Make sure the YAML file exists
        if not os.path.exists('dice_dataset.yaml'):
            create_yaml_file()

        # Initialize YOLOv8 model
        model = YOLO('yolov8n.pt')

        # Train the model
        results = model.train(
            data='dice_dataset.yaml',
            epochs=10,
            imgsz=832,
            batch=16,
            name='train'
        )

        print("Training completed successfully")
        return True

    except Exception as e:
        print(f"Error during training: {e}")
        return False


def main():
    """Main function to run the application"""
    # Check if the model exists, if not offer to run training
    if not os.path.exists(MODEL_PATH):
        print("Trained model not found.")
        response = input("Do you want to split the dataset and train a new model? (y/n): ")

        if response.lower() == 'y':
            # Split dataset
            split_data()

            # Train model
            print("Starting model training. This may take a while...")
            success = train_model()

            if not success:
                print("Training failed. Please check errors and try again.")
                sys.exit(1)
        else:
            print("Cannot continue without a trained model. Exiting.")
            sys.exit(1)

    # Start the application
    root = tk.Tk()
    app = DiceGameApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()