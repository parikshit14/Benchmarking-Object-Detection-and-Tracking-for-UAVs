# Getting started with the code base

This readme provides the basic overview of how the code looks like. The entire codebase has been divided into two main parts:
1. Training
2. Inference

Both have separate instructions, although dependencies are mostly same so having one python virtual environment should work.
(Detailed instruction on creating environment is present in the `training` folder)

## 📚 Table of Contents
- [Getting started with the code base](#getting-started-with-the-code-base)
  - [📚 Table of Contents](#-table-of-contents)
  - [📌 Project Overview](#-project-overview)
    - [Object Detection](#object-detection)
    - [Tracking Algorithms](#tracking-algorithms)
  - [✅ System Requirements](#-system-requirements)
  - [📁 Project Structure](#-project-structure)
    - [Additional Documentation](#additional-documentation)
    - [Extra](#extra)
  - [📬 Feedback](#-feedback)

## 📌 Project Overview
### Object Detection
Object detection is a computer vision technique that identifies and locates objects within digital images and videos. It goes beyond simple image recognition by not only identifying what objects are present but also pinpointing their locations with bounding boxes.

For Object detection we experimented with three detection models:
1. [**EdgeYOLO**](https://github.com/LSH9832/edgeyolo/blob/main/README_EN.md): Specifically designed loss function improves the model's precision when detecting small object; also utilizes an anchor-free strategy,  educing post-processing computational complexity
2. [**CEASC**](https://github.com/Cuogeihong/CEASC.git): Uses sparse convolutions to reduce computational complexity by focusing computations on foreground areas, reducing computational cost
3. [**Yolov8**](https://github.com/ultralytics/ultralytics): Achieves a strong balance between high detection accuracy and low inference tim; has a large active developer community enabling widespread adaptation


### Tracking Algorithms
Object tracking is a computer vision technique that follows the movement of an object or multiple objects across a sequence of video frames. It aims to determine the trajectory of the object(s) over time, even when faced with challenges like changes in appearance, occlusion, or movement. 

For Object tracking we experimented with five tracking algorithms:
1. [**DeepSORT**](https://github.com/ModelBunker/Deep-SORT-PyTorch): Uses a deep appearance descriptor to enhance tracklet continuity and reduce ID switches
2. [**ByteTrack**](https://github.com/FoundationVision/ByteTrack): Similarity comparisons with existing tracklets to recover genuine objects and filter out background noise, particularly for low-score boxes
3. [**StrongSORT**](https://github.com/dyhBUPT/StrongSORT): Gaussian-Smoothed Interpolation mitigates missing detections by refining the interpolated positions of objects
4. [**SMILETrack**](https://github.com/WWangYuHsiang/SMILEtrack): Siamese network-based component that assesses the appearance similarity between objects
5. [**BoTSORT**](https://github.com/NirAharon/BoT-SORT): Uses fine-grained association metrics for better performance in dense scenes and moving camera

**Note:** In BoTSORT and SMILETrack there are ReID options available, while in trackers either it is a compulsory (DeepSORT, StrongSORT) or not present at all(ByteTrack)

## ✅ System Requirements
1. **OS**: Ubuntu 20.04 LTS (recommended)
2. **Python**: >= 3.7
3. **Conda**: [Install Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install) (recommended version: **23.7.4**)
4. **GPU**: For training at least 24 GB VRAM GPU(e.g., RTX4090) will suffice, while for inference there is no constrains although a device with GPU memory >=8 is highly recommended
5. **CUDA**: CUDA 11.X is being used for most of the project (11.6 recommended) [Install CUDA](https://developer.nvidia.com/cuda-11-6-0-download-archive)

## 📁 Project Structure
| Folder | Description |
|--------|-------------|
| `training/` | Training scripts with readme file that describes how to setup the environment to train detection model|
| `inference/` | Inference scripts with readme file that describes how inferencing works, combining both detection and tracking |
| `sample_video/` | Contains sample video to run detection and tracking |

**Note:** There is separate readme files for inference and training in their respective folder to keep this readme file uncluttered.

### Additional Documentation
- [Training](/training/readme.md)
- [Inference](/inference/readme.md)
  

### Extra
- For custom detection model and classes, the code is modularized enough to use with any detection model, just make sure the bounding box output format of the detection is in TLWH or TLBR or XCYCWH format

## 📬 Feedback

Please feel free to report issues or submit requests bug fixes via [email](mailto:parikshits@iisc.ac.in?cc=vishwajeetp@iisc.ac.in,prathore@iisc.ac.in&subject=Bug%20Report%20-%20BEL%20UAV%20Project)!

