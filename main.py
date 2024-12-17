from kivy.app import App
from kivy.uix.image import Image as KivyImage
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import torch


class ObjectDetectionApp(App):
    def build(self):
        self.layout = BoxLayout(orientation="vertical")
        
        # Add a label for instructions
        self.label = Label(text="Click 'Select Image' to choose an image for object detection", size_hint_y=None, height=30)
        self.layout.add_widget(self.label)

        # Add a button to open the file chooser
        self.select_button = Button(text="Select Image", size_hint_y=None, height=50)
        self.select_button.bind(on_press=self.open_filechooser)
        self.layout.add_widget(self.select_button)
        
        # Add a button to run detection on the selected image
        self.detect_button = Button(text="Detect Objects", size_hint_y=None, height=50)
        self.detect_button.bind(on_press=self.run_detection)
        self.layout.add_widget(self.detect_button)
        
        # Add an image widget to display the output
        self.image_widget = KivyImage(size_hint=(1, None), height=500)
        self.layout.add_widget(self.image_widget)
        
        self.selected_image_path = None  # Store the selected image path
        return self.layout

    def open_filechooser(self, instance):
        # Open file chooser to select an image
        filechooser = FileChooserIconView()
        filechooser.filters = ['*.jpg', '*.png', '*.jpeg']  # Filter for image files
        filechooser.bind(on_selection=self.on_file_selected)

        # Popup for file chooser
        popup = Popup(title="Select Image", content=filechooser, size_hint=(0.9, 0.9))
        popup.open()

    def on_file_selected(self, filechooser, selected):
        # Get the path of the selected file
        if selected:
            self.selected_image_path = selected[0]  # Get the first selected file
            self.label.text = f"Selected: {self.selected_image_path}"

    def run_detection(self, instance):
        if not self.selected_image_path:
            self.label.text = "Please select an image first."
            return

        # Load YOLOv8 model
        model = YOLO("./models/yolov8n.pt")  # Load the YOLOv8n model from the file

        # Run YOLOv8 inference on the selected image
        results = model(self.selected_image_path, conf=0.2)  # Set confidence threshold
        
        # Process and draw detections on the image
        output_image = self.draw_detections(self.selected_image_path, results)
        
        # Convert the PIL Image to a texture for Kivy to display
        self.image_widget.texture = self.pil_to_texture(output_image)

    def draw_detections(self, image_path, results):
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(r"D:\Python Projects\Object Detection\backend\static\roboto.ttf", 18)

        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                box = box.tolist()  # Convert bounding box tensor to a list
                confidence = conf.item() * 100  # Confidence score
                label = result.names[int(cls.item())]  # Class label

                # Draw bounding box
                draw.rectangle(box, outline="red", width=3)
                text = f"{label} {confidence:.2f}%"
                draw.text((box[0], box[1]),  text, fill="white", font=font)

        return image

    def pil_to_texture(self, pil_image):
        """Convert a PIL Image to Kivy Texture"""
        from kivy.graphics.texture import Texture
        from io import BytesIO
        
        buf = BytesIO()
        pil_image.save(buf, format="png")
        buf.seek(0)
        
        texture = Texture.create(size=pil_image.size, colorfmt='rgb')
        texture.blit_buffer(buf.read(), colorfmt='rgb', bufferfmt='ubyte')
        
        return texture


if __name__ == '__main__':
    ObjectDetectionApp().run()
