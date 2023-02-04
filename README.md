
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/AlperenEnesBayar/Stroma-Take-Home-Task">
    <img src="github_imgs/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Yolov7 Detection and ByteTrack Track Pipeline</h3>

  <p align="center">
    Bold and Nut detection and tracking process for Stroma TakeHome Task
    <br />
    <br />
    <a href="https://youtu.be/6UJZD3x_8hU">View Demo</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

![res](github_imgs/main.png?raw=true "Test Results") 

Real-time object detection is a very important task that is often a key component in computer vision systems. Applications that use real-time object detection models include video analytics, robotics, autonomous vehicles, multi-object tracking and object counting, medical image analysis, and so on.

YOLO stands for “You Only Look Once”, it is a popular family of real-time object detection algorithms. The original YOLO object detector was first released in 2016. It was created by Joseph Redmon, Ali Farhadi, and Santosh Divvala. At release, this architecture was much faster than other object detectors and became state-of-the-art for real-time computer vision applications

YOLOv7 is the fastest and most accurate real-time object detection model for computer vision tasks. The official YOLOv7 paper named “YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors” was released in July 2022 by Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao.

Here's why:
* Extended Efficient Layer Aggregation Network (E-ELAN)
* Model Scaling for Concatenation based Models
* Planned re-parameterized convolution
* Trainable Bag of Freebies
* Planned re-parameterized convolution
* Coarse for auxiliary and fine for lead loss

For the tacking part I use ByteTrack.

A simple, effective, and a generic association method to track objects by associating almost every detection box instead of just the high score ones

BYTE outperforms other association methods like SORT, Deep SORT, and MOTDT by a large margin

ByteTrack is a simple yet effective algorithm for multi-object tracking(MOT). It uses YOLOX, a high-performance object detector, and BYTE for data association. BYTE uses all of the detection results, both low and high detection scores, to enhance the performance of ByteTrack. ByteTrack is robust to occlusion, motion blur, and size changes and performs accurate tracking.

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.
                
                                                                                   
### Installation                                                                   
                                                                                   
* Clone the repo                                                                   
   ```sh                                                                           
   git clone https://github.com/AlperenEnesBayar/Stroma-Take-Home-Task.git         
   ```                                                                             

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* Anaconda
  ```sh
  conda enc create -f stroma.yaml
  ```                            
* pip
  ```sh
  pip install -r requirements.txt
  ``` 

<!-- USAGE EXAMPLES -->
## Usage
* Go to project path
   ```sh  
    python detectNtrack.py --weights trained_models/best.pt --source test.mp4      
   ```  

## Some Training Metrics
### Confusion Matrix
![conf](github_imgs/confusion_matrix.png?raw "Confusion Matrix")

### F1 Curve
![f1](github_imgs/F1_curve.png?raw=true "F1")

### P Curve
![p](github_imgs/P_curve.png?raw=true "P Curve")

### R Curve
![r](github_imgs/R_curve.png?raw=true "R Curve")

### PR Curve
![pr](github_imgs/PR_curve.png?raw=true "PR Curve")

### Some Test Results
![res](github_imgs/test.png?raw=true "Test Results")


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
