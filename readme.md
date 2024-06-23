# YOLOv10n Implementation on Driver Drowsiness Detection

An implementation of [Ultralytics - YOLOv10](https://github.com/THU-MIG/yolov10) repository for training on blink and yawn datasets to detect driver drowsiness.

## Table of Contents
- [About](#about)
- [Installation](#installation)
- [Usage](#usage)

## About
This inference repository uses the transfer learning method from EfficientDet-D0 with the following training configuration:
- YOLOv8 Version : n version (smallest)
- Image input size: 512
- Learning rate: 0,001 (1e-3)
- Batch size: 16
- Epochs: 25
- Dataloader workers: 2 

The training was stopped at 25 epochs due to an increasing total loss (it's unclear if this is relevant). The three latest epochs of the training progress resulted in the following code snippets: (TBA)


The final validation from the model was tested with 2903 images (20% of the dataset) and resulted in the following metrics: 
```py
                 Class     Images  Instances      Box(P          R      mAP50    mAP50-95)
                   all       2903       1031      0.904      0.924      0.972      0.741
           closed-eyes       2903        598      0.903       0.93      0.969      0.684
                  yawn       2903        433      0.905      0.919      0.975      0.798
```

## Installation
1. Clone this GitHub repository:
```bash
git clone https://github.com/radityamuhammadf/YOLOv10n-Implementation-on-Driver-Drowsiness-Detection.git
```
2. Change the current working directory (cwd):
```bash
cd YOLOv10n-Implementation-on-Driver-Drowsiness-Detection
```
3. Create a virtual environment:
```sh
python -m venv venv
```
4. Activate virtual environment:
```sh
If Using Windows Command Prompt:
venv\Scripts\activate

If Using Linux Bash Terminal:
source venv/bin/activate
```
5. Install all the dependencies and PyTorch
Note: Ensure your PyTorch configuration matches your CUDA version (e.g., if you're using CUDA 12.5, consider using [PyTorch for CUDA 12.4](https://pytorch.org/) ) 
```sh
pip install -r requriements.txt
```
Externally configure the YOLOv10 Library and Other Dependencies
```sh
# Import and Installing YOLOv10 Library via Cloning Repository and Other Dependencies
pip install supervision git+https://github.com/THU-MIG/yolov10.git huggingface_hub openpyxl

# Because somehow the repository contains opencv-headless, youll need to reinstall opencv by removing the headless opencv first and then reinstalling the opencv python with this command:
pip uninstall opencv-python-headless opencv-python opencv-contrib-python
pip install opencv-python
```

## Usage
Run the inference code
1.  From the video input (could be accessed but still in development)
a. If you're using Windows, then:
```sh
py video_input.py
```
b. If you're using Jetson or any other Linux Device, then:
```sh
py jetson_video_input.py
```
2.  Live Detection
```sh
py live_detection.py
```
**Additional Information**
Some modules may not be listed in the requirements.txt file (it is unclear why they are missing even after using the pip freeze command). If you encounter a `ModuleNotFoundError`, you can install the missing module using the pip command.

