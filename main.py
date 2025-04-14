import sys, os
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, QLineEdit, QPushButton, QMessageBox, QHBoxLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import QDialog, QGridLayout, QPushButton, QScrollArea, QWidget
from PyQt6.QtCore import QSize, pyqtSignal
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array


if not os.path.exists("Model.keras"):
    print("Model file not found!")
else:
    print("yes")


class ImageGallery(QDialog):
    character_selected = pyqtSignal(str)  # Signal to emit the predicted character
    def __init__(self, target_images_folder, model_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Gallery")
        self.setGeometry(100, 100, 800, 600)

        self.target_images_folder = target_images_folder
        self.model = load_model(model_path)  # Load the Keras model

        # Layout for the gallery
        layout = QGridLayout()
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)



        # Load images from the folder
        images = [f for f in os.listdir(self.target_images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for i, image_name in enumerate(images):
            image_path = os.path.join(self.target_images_folder, image_name)
            button = QPushButton()
            button.setIcon(QIcon(QPixmap(image_path).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)))
            button.setIconSize(QSize(100, 100))
            button.clicked.connect(lambda _, path=image_path: self.classify_image(path))
            scroll_layout.addWidget(button, i // 4, i % 4)  # Arrange buttons in a grid

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        self.setLayout(layout)

    def classify_image(self, image_path):
        """Classify the selected image using the Keras model."""
        # Preprocess the image
        image = load_img(image_path, target_size=(331, 331))  # Adjust size to match your model's input
        image_array = img_to_array(image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict using the model
        prediction = self.model.predict(image_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]

        # Map the predicted class index to the class name
        class_names = ["Anakin", "Boba_Fett", "C3PO", "Chewbacca", "Darth_Vader", 
                       "Leia", "Luke", "Mace_Windu", "Palpatine", "Yoda"]
        predicted_class_name = class_names[predicted_class_index]

        # Emit the predicted character and close the dialog
        self.character_selected.emit(predicted_class_name)
        self.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        screen_dims = QApplication.primaryScreen().availableGeometry()

        width = screen_dims.width()
        height = screen_dims.height()

        # Load the Keras model
        self.model = load_model("Model.keras")

        self.setWindowTitle("Star Wars Character Classifer")
        self.setGeometry((width-800)//2, (height-600)//2, 800, 600)

        self.setStyleSheet("""
            QMainWindow {
                background-image: url("Background_GUI.jpg");
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }
        """)

        self.target_images_folder = os.path.join(os.getcwd(), "target_images")
        
        if not os.path.exists(self.target_images_folder):
            os.makedirs(self.target_images_folder)

        # OpenCV video capture
        self.cap = cv2.VideoCapture(0)  # 0 = default webcam

        self.timer = QTimer()

        # Create a label to display the video
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(640, 480)

        # drag and drop file input
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Drag and drop an image file here or enter the file path...")
        self.file_input.setFixedHeight(30)
        self.file_input.setFixedWidth(300)
        self.file_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_input.setAcceptDrops(True)  # Enable drag-and-drop
        self.file_input.dragEnterEvent = self.drag_enter_event
        self.file_input.dropEvent = self.drop_event

        ## Guessed character label
        self.guessed_character_label = QLabel("")
        self.guessed_character_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.guessed_character_label.setStyleSheet("font-size: 24px; color: white; font-family: 'Impact';")

        # Back button
        self.back_button = QPushButton("Back")
        self.back_button.setFixedSize(120, 40)
        self.back_button.clicked.connect(self.restore_ui)
        self.back_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.back_button.hide()  # Initially hide the Back button

        # --- Show Images Button ---
        self.show_images_button = QPushButton("Show Images")
        self.show_images_button.setFixedSize(120, 40)
        self.show_images_button.clicked.connect(self.open_image_gallery)
        self.show_images_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Create a grid layout for the image gallery
        self.image_grid_layout = QGridLayout()
        self.image_grid_widget = QWidget()
        self.image_grid_widget.setLayout(self.image_grid_layout)
        self.image_grid_widget.hide()  # Initially hide the image grid


        # --- Camera button ---
        self.camera_button = QPushButton("Open Camera")
        self.camera_button.setCheckable(True)
        self.camera_button.clicked.connect(self.launch_camera)
        self.camera_button.setFixedSize(120, 40)
        self.camera_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Instructions
        self.instructions1 = QLabel("Enter Image Name")
        self.instructions1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instructions1.setStyleSheet("font-size: 24px; color: yellow; font-family: 'Impact';")

        self.instructions2 = QLabel("OR")
        self.instructions2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instructions2.setStyleSheet("font-size: 24px; color: yellow; font-family: 'Impact';")

        self.instructions3 = QLabel("Take Picture")
        self.instructions3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instructions3.setStyleSheet("font-size: 24px; color: yellow; font-family: 'Impact';")

        # Set layout
        self.layout = QVBoxLayout()
        self.layout.addStretch()   
        self.layout.addWidget(self.instructions1)
        self.layout.addWidget(self.file_input, alignment=Qt.AlignmentFlag.AlignHCenter)  # File input bar at the top

        # Add the "Enter" button
        self.enter_button = QPushButton("Enter")
        self.enter_button.setFixedSize(120, 40)
        self.enter_button.clicked.connect(self.save_image_from_path)
        self.enter_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Add the image grid widget to the main layout
        self.layout.addWidget(self.image_grid_widget)

        # Add the "Enter" button to the layout
        self.layout.addWidget(self.enter_button, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.layout.addStretch()  
        self.layout.addWidget(self.instructions2)
        self.layout.addStretch()   
        self.layout.addWidget(self.instructions3)
        self.layout.addWidget(self.camera_button, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.layout.addWidget(self.guessed_character_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.layout.addStretch()

        # Add show button to the layout
        self.layout.addWidget(self.show_images_button, alignment=Qt.AlignmentFlag.AlignHCenter)



        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def drag_enter_event(self, event):
        """Handle drag enter event to accept file drops."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def save_image_from_path(self):
        """Save the image from the file path entered in the input field."""
        file_path = self.file_input.text()
        if os.path.exists(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Copy the image to the target_images folder
            target_path = os.path.join(self.target_images_folder, os.path.basename(file_path))
            cv2.imwrite(target_path, cv2.imread(file_path))
            QMessageBox.information(self, "Success", f"Image saved to {target_path}")
        else:
            QMessageBox.warning(self, "Error", "Invalid file path or unsupported file format.")

    def drop_event(self, event):
        """Handle drop event to get the file path."""
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.file_input.setText(file_path)

    def open_image_gallery(self):
        """Display the image gallery in the main window."""
        # Clear the grid layout
        for i in reversed(range(self.image_grid_layout.count())):
            widget = self.image_grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Load images from the target_images folder
        images = [f for f in os.listdir(self.target_images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for i, image_name in enumerate(images):
            image_path = os.path.join(self.target_images_folder, image_name)
            button = QPushButton()
            button.setIcon(QIcon(QPixmap(image_path).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)))
            button.setIconSize(QSize(100, 100))
            button.clicked.connect(lambda _, path=image_path: self.classify_image(path))
            self.image_grid_layout.addWidget(button, i // 4, i % 4)  # Arrange buttons in a grid

        self.image_grid_layout.addWidget(self.back_button, len(images) // 4 + 1, 0, 1, 4)  # Span across the grid width
        # Hide other widgets and show the image grid
        self.instructions1.hide()
        self.instructions2.hide()
        self.instructions3.hide()
        self.file_input.hide()
        self.camera_button.hide()
        self.show_images_button.hide()
        self.enter_button.hide()
        self.guessed_character_label.hide()
        self.image_grid_widget.show()
        self.back_button.show()  # Show the Back button

    def classify_image(self, image_path):
        """Classify the selected image using the Keras model."""
        # Preprocess the image
        image = load_img(image_path, target_size=(331, 331))  # Adjust size to match your model's input
        image_array = img_to_array(image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict using the model
        prediction = self.model.predict(image_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]

        # Map the predicted class index to the class name
        class_names = ["Anakin", "Boba_Fett", "C3PO", "Chewbacca", "Darth_Vader", 
                       "Leia", "Luke", "Mace_Windu", "Palpatine", "Yoda"]
        predicted_class_name = class_names[predicted_class_index]

        # Show the guessed character
        self.show_guessed_character(predicted_class_name)

    def restore_ui(self):
        """Restore the original UI after showing the guessed character."""
        self.instructions1.show()
        self.instructions2.setText("OR")
        self.instructions2.setStyleSheet("font-size: 24px; color: yellow; font-family: 'Impact';")
        self.instructions3.show()
        self.file_input.show()
        self.camera_button.show()
        self.show_images_button.show()
        self.guessed_character_label.show()
        self.enter_button.show()  # Ensure the Enter button reappears
        self.image_grid_widget.hide()  # Hide the image grid

    def show_guessed_character(self, character_name):
        """Display the guessed character in the entire window."""
        # Hide all other widgets
        self.instructions1.hide()
        self.instructions2.hide()
        self.instructions3.hide()
        self.file_input.hide()
        self.camera_button.hide()
        self.show_images_button.hide()
        self.guessed_character_label.hide()
        self.enter_button.hide()  # Hide the Enter button

        # Create a temporary label to display the guessed character
        self.instructions2.setText(f"GUESS: {character_name}")
        self.instructions2.setStyleSheet("font-size: 48px; color: white; font-family: 'Courier';")
        self.instructions2.show()

        # Show the guessed character for 2 seconds, then restore the UI
        QTimer.singleShot(2000, lambda: self.restore_ui())
            
    def launch_camera(self):
        if self.camera_button.isChecked():
            self.camera_button.setText("Close Camera")
            self.instructions1.hide()
            self.instructions2.hide()
            self.file_input.hide()
            self.enter_button.hide()
            self.show_images_button.hide()
            self.instructions3.setText("PRESS SPACE TO CAPTURE IMAGE")
            self.layout.insertWidget(1, self.image_label, alignment=Qt.AlignmentFlag.AlignHCenter)  # Show camera feed
            self.image_label.show()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)
            self.setFocus()
            self.image_label.setStyleSheet("border: 10px solid yellow; border-radius: 10px;")
        else:
            self.timer.stop()
            self.image_label.clear()
            self.image_label.hide()
            self.instructions1.show()
            self.instructions2.show()
            self.file_input.show()
            self.enter_button.show()
            self.show_images_button.show()
            self.instructions3.setText("Take Picture")
            self.camera_button.setText("Open Camera")
            self.image_label.setStyleSheet("")


    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            ))


    def closeEvent(self, event):
        # Release the capture when closing
        self.cap.release()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and self.camera_button.isChecked():
            ret, frame = self.cap.read()
            if ret:
                # Save the image to the target_images folder
                image_path = os.path.join(self.target_images_folder, "captured_image.jpg")
                cv2.imwrite(image_path, frame)
                print(f"Image saved to {image_path}")
                time.sleep(1)

            self.timer.stop()
            self.image_label.clear()     # Optional: clear the last frame
            self.image_label.hide()
            self.instructions3.hide()
            self.camera_button.hide()

            # Give User Confirmation
            self.instructions2.setText("IMAGE SAVED")
            self.instructions2.setStyleSheet("font-size: 32px; color: white; font-family: 'Courier';")
            self.instructions2.show()

            # Delay showing instructions and file input after 1.5 seconds
            QTimer.singleShot(1000, lambda: (
                self.instructions3.setText("Take Picture"),
                self.instructions3.setStyleSheet("font-size: 24px; color: yellow; font-family: 'Impact';"),
                self.instructions2.setText("OR"),
                self.instructions2.setStyleSheet("font-size: 24px; color: yellow; font-family: 'Impact';"),
                self.instructions1.show(),
                self.file_input.show(),
                self.instructions3.show(),
                self.camera_button.show(),
                self.enter_button.show(),
                self.show_images_button.show()
            ))

            self.camera_button.setChecked(False)
            self.camera_button.setText("Open Camera")



if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())





