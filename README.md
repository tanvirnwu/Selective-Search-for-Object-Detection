# **Object Detection Using Selective Search**

This repository provides a Python implementation of Selective Search for object detection. It includes code for segmenting an image using the Felzenszwalb method, extracting candidate regions for object detection, and visualizing these regions.

Selective Search is a region proposal algorithm used in object detection. It is designed to identify potential bounding boxes in an image that are likely to contain objects. This technique combines the strength of both exhaustive search and segmentation. As opposed to traditional methods that look at thousands of random regions, Selective Search focuses on those parts of the image that have a higher likelihood of forming meaningful configurations, making it both efficient and powerful for object detection tasks.

### Prerequisites

Make sure you have Python installed on your system. You can download Python [here](https://www.python.org/downloads/). Then, install the required packages using the following commands:

```bash
pip install selectivesearch
pip install torch_snippets
pip install opencv-python-headless # Use opencv-python if you need GUI components
```

### Libraries Used
- selectivesearch
- torch_snippets
- cv2
- matplotlib
- numpy
- skimage

### Thanks for reading!!
