import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, QLineEdit, QPushButton
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
import time

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        screen_dims = QApplication.primaryScreen().availableGeometry()

        width = screen_dims.width()
        height = screen_dims.height()

        self.setWindowTitle("Star Wars Character Classifer")
        self.setGeometry((width-800)//2, (height-600)//2, 800, 600)

        self.setStyleSheet("""
            QMainWindow {
                background-image: url('Background_GUI.jpg');
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }
        """)

        # OpenCV video capture
        self.cap = cv2.VideoCapture(0)  # 0 = default webcam

        self.timer = QTimer()

        # Create a label to display the video
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(640, 480)

         # --- File path input bar ---
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Enter image file path...")
        self.file_input.setFixedHeight(30)
        self.file_input.setFixedWidth(300)
        self.file_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

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
        self.layout.addStretch()                # Space between input bar and button
        self.layout.addWidget(self.instructions2)
        self.layout.addStretch()   
        self.layout.addWidget(self.instructions3)
        self.layout.addWidget(self.camera_button, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.layout.addStretch()

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)


    def launch_camera(self):
        if self.camera_button.isChecked():
            self.camera_button.setText("Close Camera")
            self.instructions1.hide()
            self.instructions2.hide()
            self.file_input.hide()
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
                cv2.imwrite("captured_image.jpg", frame)
                print("Image saved!")
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
                self.camera_button.show()
            ))

            self.camera_button.setChecked(False)
            self.camera_button.setText("Open Camera")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())









