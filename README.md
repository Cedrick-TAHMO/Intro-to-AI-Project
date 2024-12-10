Welcome to my weapon detection program powered by YOLOv8

Dataset: https://drive.google.com/drive/folders/1LeXLIr1JIMCCf3zMONAFruAshp6SvtDW?usp=drive_link

Steps on how to achieve this in Google Colab

Step 1: We must change the runtime environment from CPU to GPU. 

Runtime > Change runtime Type
        
Step 2: Type the following check that you are indeed on the GPU runtime

    !nvidia-smi

Step 3: Install the Ultralytics Python package, which includes the YOLOv8, YOLOv5 framework. 
It is necessary for training, validating, and running inferences using YOLO models

    pip install ultralytics

Step 4:
Imports the YOLO class from the Ultralytics package, allowing you to load and work with YOLO models.
import os: Imports Python's os module, which is used for handling file and directory operations.
from IPython.display import display, Image: Imports functions to display images and outputs in a Jupyter Notebook.
from IPython import display and display.clear_output(): Clear previous outputs in a Jupyter Notebook for a cleaner presentation.

    from ultralytics import YOLO
    import os
    from IPython.display import display, Image
    from IPython import display
    display.clear_output()

Step 5: This code loads the dataset from a local file. The dataset can be obtained from the link and uploaded to your Colab workspace.
The last piece of code will return a True if the file exists and the file path is correct. 

    import os
    from ultralytics import YOLO
      
    # Path to your local dataset
    dataset_path = "/content/Weapons dataset.v2i.yolov5pytorch/data.yaml"
      
    # Load the YOLOv5 model
    model = YOLO("yolov5nu.pt")  # Replace with your specific YOLO model file
      
    file_path = "/content/Weapons_dataset.v2i.yolov5pytorch/data.yaml"
    print(os.path.exists(file_path))  # Should return True if the file exists


!yolo: Runs the YOLO command-line interface in a Jupyter Notebook or similar environment.

task=detect: Specifies the task type as object detection.

mode=train: Indicates the training mode to fine-tune the model on a custom dataset.

model=yolov5nu.pt: Specifies the pre-trained YOLO model to use as the starting point (yolov5nu.pt).

data="/content/Weapons_dataset.v2i.yolov5pytorch/data.yaml": Points to the dataset configuration file (usually containing class labels and paths to training/validation data).

epochs=30: Sets the number of training epochs (iterations over the dataset) to 30.

imgsz=640: Defines the image size (640x640 pixels) for training and inference.

Step 6: This command trains the YOLO model on a custom weapon detection dataset for 30 epochs:

    !yolo task=detect mode=train model=yolov5nu.pt data="/content/Weapons_dataset.v2i.yolov5pytorch/data.yaml" epochs=30 imgsz=640

Step 7: Run these lines of code to see the statistics
    
    from IPython.display import Image

    image_path = '/content/runs/detect/train11/results.png'
    Image(filename=image_path, width=1000)  # Width can be adjusted as desired

Step 8: Run the code to test the model with random images downloaded online. 
Note: Feel free to download images from the web and test them with the model
    
    from ultralytics import YOLO

    # Load your trained model
    model = YOLO('/content/runs/detect/train11/weights/best.pt')
    
    # Path to your image
    image_path = '/content/Weapons_dataset.v2i.yolov5pytorch/OnlineImg/test3.jpg'
    
    # Perform inference
    results = model(image_path)
    
    # Display results
    for result in results:
        result.show()
    result.save(filename="result.jpg")

Step 9: Run code to test the model with random videos obtained online
Note: Note: Feel free to download videos from the web and test them with the model

    from ultralytics import YOLO

    # Load your trained model
    model = YOLO('/content/runs/detect/train/weights/best.pt')
    
    # Path to your video
    video_path = '/content/sample_data/test4.mp4'
    
    # Perform inference
    results = model(video_path, stream=True)
    
    # Process results
    for result in results:
        boxes = result.boxes
        result.show()
        result.save(filename="result_frame.jpg")
