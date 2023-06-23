# Tiny-TimeNAS, KAIST CS565 Mini-Project (Spring 2023)
> A Python Package for Time-Series Analysis on IoT Devices with Tiny Neural Architecture Search

Author: Patara Trirat (patara.t@kaist.ac.kr) [[Final Paper](/paper.pdf)] [[Final Presentation](https://docs.google.com/presentation/d/1PvrrmgyCk8nw9__2zsZdBYRM7sAJUgTYgGv-WpB4wAg/edit#slide=id.g2294fc388e6_0_11)]

![](/figures/idea_overview.jpg?raw=true "Tiny-TimeNAS")


## Introduction Video
[![Introduction Video](https://img.youtube.com/vi/X-qmMkeJCng/0.jpg)](https://www.youtube.com/watch?v=X-qmMkeJCng)


## Project Directory
```bash
.
├── datasets # folder containing training and testing datasets for architecture search and evaluation
│   ├── anomaly_detection # anomaly detection task datasets
│   ├── classification # classification task datasets
│   └── regression # regression task datasets
├── demo_search.ipynb # a step-by-step demonstration notebook of Tiny-TimeNAS (both classification and regression)
├── LICENSE
├── TinyTNAS_Classification.ino # main sketch file for deployment classification task (example)
├── models # folder for saving converted models
├── requirements.txt # necessary dependencies
└── tiny_tnas # main package folder
    ├── converters.py # function for TF Lite conversion
    ├── data_loader.py # data loader and preprocessing interface
    ├── evaluator.py # collective evaluation metrics for each task
    ├── profiler.py # pre training TFLite model profiling
    ├── searcher.py # main search method
    ├── search_space.py # main search space interface and simple networks
    └── zero_proxies.py # interface for zero-cost proxies
```

## Installation
For **architecture search** phase, run the following command for dependencies installation (highly recommended with Python 3.9 on a virtual environment or conda environment). Then, follow the `demo_search.ipynb` notebook file.
```bash
git clone https://github.com/Kaist-ICLab/final-submission-team-11-20205642.git
cd final-submission-team-11-20205642

pip install -r requirements.txt
```

For **model deployment** phase, install the following [`Arduino_TensorFlowLite`](https://github.com/tensorflow/tflite-micro-arduino-examples) in your Arduino libraries and any libraries relevant to your specific applications (e.g., `Arduino_LSM9DS1` for IMU-based classification). Then, open the `TinyTNAS_Classification.ino` sketch file.
```bash
# in, for example, My Documents\Arduino\libraries
git clone https://github.com/tensorflow/tflite-micro-arduino-examples Arduino_TensorFlowLite
cd Arduino_TensorFlowLite
git pull
```


## Example Usage
1. After the above installation, first follow the `demo_search.ipynb` notebook file with *classification* task on *BasicMotions* dataset to run architecture search, evaluation, and conversion. Please note that you can also use your own dataset(s) by simply providing **x_train** and **y_train** for both classification and regression tasks.
2. Then, download the `models/model.cpp`, `model.h`, and `TinyTimeNAS_Classification.ino` to your **Arduino Sketchbook** folder.
3. Upload the downloaded codes to the device.
4. Test with the Sensing & TinyML in real-time. In this example, you can observe the detection results via *serial monitor* and *LED* feedback.
