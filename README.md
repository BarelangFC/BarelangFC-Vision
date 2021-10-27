# Welcome to the BarelangFC-Vision
BarelangFC robot vision is using a yolo deep learning to get a custom object detection like to detect ball, goal, and other obstacle
To get a result the vision we must do this following step : 
1. Install all requirements and library
1. Get image data
1. Train / learning data
1. Demo / using data to robot

## 1. Install all requirements and library
First you must install all requairements and dependencies on this [link](https://robocademy.com/2020/05/01/a-gentle-introduction-to-yolo-v4-for-object-detection-in-ubuntu-20-04/) like:
- CMake
- Python
- Cuda
- OpenCV
- cuDNN
- ZED SDK (if using ZED Camera) to install sdk you can get from this [link](https://www.stereolabs.com/developers/release/)

## 2. Get image data
Get data image using python file getData.py run the code
``` 
python getData.py 
```
As a note _IMAGE_RGB_PATH_ is the path the image saved on your computer and Yolo Mark to do labelling or marking the custom object you can get:
```
git clone https://github.com/AlexeyAB/Yolo_mark.git
```
Put the previously captured image into the _x64/Release/data.img_ directory, don't forget to change obj.names according to what you want to mark and do this :
```
cd Yolo_mark
```
```
cmake .
```
```
make
```
```
chmod +x ./linux_mark.sh
```
After that mark the image according to the specified obj.names, don't get it wrong

## 3. Train / learning data
```
./darknet detector train obj.data yolov3-tiny_2021.cfg darknet53.conv.74
```
It's for learning image _yolov3-tiny_2021.cfg_ is a custom our config file
```
./darknet detector calc_anchors obj.data -num_of_clusters 9 -width 416 -height 416 
```
It's for calc anchors

## 4. Demo / using data to robot
After install all requirements you can get or clone from 
``` 
git clone https://github.com/AlexeyAB/darknet.git
```
* and next prepare data in yolo_mark
* copy img, obj.data, obj.names, train.txt
* paste obj.data in the darknet/cfg folder
* paste obj.names and train.txt in the darknet/data folder
* create x64/Release/data folder. paste img
* create a new cfg file(ex: yolov3-tiny_tes.cfg). copy paste the contents of yolov3-tiny.cfg
* change batch = 64
* change subdivision = 8 //the smaller the gpu usage the bigger the learning is
* change classes = 2 (because there are only 2, namely ball and goal, adjust to the number of classes marked) because of yolov3, so filter= (classess+5)*3. so filter =21
* changed filters only on lines 127 and 171
```
./darknet detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416
```
copy the result, paste it into the anchor in the cfg
```
./darknet detector train cfg/obj.data cfg/yolov3-tiny_tes.cfg darknet53.conv.74
```
All the command text you can create custom files.sh as you want, thank you that's enough from us and enjoy to learn.

