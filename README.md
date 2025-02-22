Semantic Segmentation using Onnx with OpenCV </br>
Segments aerial UAV imagery </br>
Model has 8 labels, takes an image and classifies each pixel with a label </br>
if no classification, pixels are automatically given 0 </br>
labels start from 1 </br>

run using 
```
./aerial_segment <test image> <onnx model> <labels.txt> <--use_cuda|--use_cpu> <sigmoid threshold (0.8)>
```
</br>

Created Evan O'Keeffe with MIT license
