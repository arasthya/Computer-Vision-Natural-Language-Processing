# Computer-Vision-Natural-Language-Processing
Team task at Startup Campus (Artificial Intelligence Track)

---

### Case 1: Smartphone Image Enhancement with Digital Image Processing

#### Overview
This project demonstrates digital image processing techniques commonly used in smartphone cameras to enhance images, particularly in low-light conditions. By exploring methods like Max Pooling and Contrast Limited Adaptive Histogram Equalization (CLAHE), this project aims to replicate techniques similar to those utilized by top smartphone brands such as Apple and Samsung to improve photo quality.

#### Dataset
This project uses two sample images, **photo1.jpeg** and **lena.png**. Ensure both images are uploaded to the working directory, as they serve as the primary images for testing image processing functions.

#### Objectives
- Understand and apply basic image processing techniques for low-light enhancement.
- Explore the concept of **Max Pooling** and its application in image brightening.
- Compare Max Pooling with **CLAHE** to examine the effectiveness of different enhancement techniques.
- Save enhanced images as part of the output to visualize the differences in enhancement methods.

#### Analysis & Modelling Tools
- **OpenCV**: Used for basic image manipulation, color conversion, and histogram analysis.
- **Scikit-image**: Supports block-based image processing, specifically Max Pooling with custom block sizes.
- **PyTorch**: Used to perform GPU-accelerated Max Pooling operations, allowing for efficient processing on large images.
- **Matplotlib**: Facilitates the visualization of original and processed images, as well as histograms for color and grayscale images.
- **NumPy**: Utilized for efficient numerical computations and array manipulations.

---

### Case 2: Transfer Learning with Pre-trained CNN

##### Overview
This project demonstrates the use of transfer learning to perform image classification using pre-trained Convolutional Neural Network (CNN) models. By leveraging knowledge from models initially trained on a large dataset (ImageNet), we aim to improve classification performance on our target dataset with a reduced training time.

#### Dataset
All models used in this project were originally trained on the ImageNet dataset, containing millions of RGB images (3 channels) spanning 1000 classes.

#### Objectives
- To explore the effectiveness of transfer learning for image classification.
- To evaluate model performance with frozen layers vs. fine-tuning on the target dataset.
- To compare different CNN architectures such as ResNet, DenseNet, and ViT.

#### Analysis & Modelling Tools
- **Libraries**: TensorFlow-Keras, PyTorch
- **Models**: Pre-trained CNN models including ResNet, DenseNet, and ViT
- **Evaluation Metrics**: Accuracy, Loss

---

### Case 3: Real-Time Object Detection Using Pre-trained CNN (YOLOv5)

#### Overview
This project demonstrates real-time object detection using a CNN-based pre-trained model, YOLOv5. The model is loaded from PyTorch's Hub and used to detect various objects in a YouTube video stream. The application processes video frames from a given YouTube URL and visualizes the detection results by drawing bounding boxes around detected objects. This is implemented using PyTorch, OpenCV, and a custom utility to capture YouTube video streams.

#### Dataset
The pre-trained model (YOLOv5) used in this project was trained on the **COCO dataset**, which contains a wide variety of common objects such as humans, animals, vehicles, etc. The model can detect and label these objects in real-time from video frames. While the COCO dataset is comprehensive, it may struggle with objects or scenes that fall outside of its typical scope, such as outer space or abstract environments. 

For this implementation, YouTube videos serve as the input, and any video URL can be used. Example videos include:
1. **Crowded place**: https://www.youtube.com/watch?v=dwD1n7N7EAg
2. **Solar system**: https://www.youtube.com/watch?v=g2KmtA97HxY
3. **Road traffic**: https://www.youtube.com/watch?v=wqctLW0Hb_0

## **Objectives**
- Implement a real-time object detection pipeline using YOLOv5 for video streams.
- Extract and process video frames from YouTube using OpenCV and a custom utility.
- Detect and classify objects in each frame of the video using a pre-trained CNN model.
- Display bounding boxes around detected objects and output the processed video.
- Implement a Python script that uses PyTorch and OpenCV to perform object detection and visualize results in real-time.

#### Analysis & Modelling Tools
- **PyTorch**: A deep learning framework that provides pre-trained models like YOLOv5. It is used to load and run the model for object detection tasks.
- **OpenCV**: A library for real-time computer vision tasks, including capturing video frames and drawing bounding boxes around detected objects.
- **YOLOv5 (You Only Look Once)**: A real-time object detection model known for its speed and accuracy, especially with common objects. The pre-trained version of YOLOv5 was used in this project, which has been trained on the COCO dataset.
- **cap-from-youtube**: A custom utility that allows easy extraction of video frames from a YouTube video stream.

---

### Case 4: Natural Language Processing - Text Classification with BERT

#### Overview
This project focuses on using Natural Language Understanding (NLU) to classify tweets related to disasters using BERT (Bidirectional Encoder Representations from Transformers). The goal is to develop a model that can accurately predict whether a tweet is related to a disaster or not. The project involves fine-tuning a pre-trained BERT model for text classification, leveraging the power of transfer learning to achieve high accuracy.

#### Dataset
The dataset used in this project consists of tweets labeled as either disaster-related or not. It includes various real-world tweets gathered from Twitter, making it a rich source of unstructured text data. The dataset is used for training, validation, and testing the model to classify disaster-related tweets.

#### Objectives
- Preprocess and clean the tweet text data for model input.
- Fine-tune a pre-trained BERT model for text classification.
- Evaluate the model’s performance in distinguishing between disaster-related and non-disaster-related tweets.
- Achieve high classification accuracy through proper tuning and transfer learning.

#### Analysis & Modelling Tools
- **Data Preprocessing**: Python libraries like Pandas and Numpy for data manipulation, and NLTK or SpaCy for text cleaning.
- **Modeling**: TensorFlow-Keras and Hugging Face’s Transformers library for fine-tuning the BERT model.
- **Evaluation**: Metrics like accuracy, precision, recall, and F1 score to evaluate model performance.

--- 
