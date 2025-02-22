// https://github.com/microsoft/onnxruntime/blob/v1.8.2/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
// https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h
// #include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <ort_session_handler.hpp>


std::vector<std::string> readLabels(std::string& labelFilepath) {
  std::vector<std::string> labels;
  std::string line;
  std::ifstream fp(labelFilepath);
  while (std::getline(fp, line)) {
    labels.push_back(line);
  }
  return labels;
}

int main(int argc, char* argv[]) {
  int inpWidth = 1024;
  int inpHeight = 576;
  float sigmoid_threshold = 0.8;

  std::vector<const char*> input_node_names = { "image" }; // Input node names
  std::vector<int64_t> input_dims = { 1, 3, inpHeight, inpWidth };
  std::vector<int64_t> output_dims = {1, inpHeight, inpWidth };
  std::vector<const char*> output_node_names = { "sigmoid" }; // Output node names
  std::vector<float> pytorch_mean = {0.485, 0.456, 0.406};
  std::vector<float> pytorch_std = {0.229, 0.224, 0.225};

  const int64_t batchSize = 1;
  bool useCUDA = false;
  const char* useCUDAFlag = "--use_cuda";
  const char* useCPUFlag = "--use_cpu";

  std::string imageFilepath{argv[1]};
  std::string modelFilepath{argv[2]};
  std::string labelFilepath{argv[3]};

  if (argc>3 && strcmp(argv[4], useCUDAFlag) == 0){
    useCUDA = true;
  }

  if(argc>4){
    sigmoid_threshold = std::atof(argv[5]);
  }

  std::cout << imageFilepath << std::endl;
  std::cout << modelFilepath << std::endl;
  std::cout << labelFilepath << std::endl;
  std::cout << sigmoid_threshold << std::endl;


  if (useCUDA){
    std::cout << "Inference Execution Provider: CUDA" << std::endl;
  }
  else{
    std::cout << "Inference Execution Provider: CPU" << std::endl;
  }

  std::vector<std::string> labels = readLabels(labelFilepath);
  
  deploy::OrtSessionHandler ort_session_handler(modelFilepath, input_dims);

  Ort::MemoryInfo memory_info{ nullptr };     // Used to allocate memory for input
  try {
    memory_info = std::move(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
  }
  catch (Ort::Exception oe) {
    std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
    return -1;
  }

  // from here we can load each image to classify

  cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
  int original_width = imageBGR.cols;
  int original_height = imageBGR.rows;

  if(imageBGR.empty()){
    std::cerr << "issue reading the image" << std::endl;
  }else{
    std::cout << "image read success" << std::endl;
  }

  // convert the cv::Mat to std::vector<float>
  std::vector<float> input_data = ort_session_handler.preprocess(imageBGR, inpHeight, inpWidth, pytorch_mean, pytorch_std);

  // push the std::vector<float> into Tensor
  std::vector<Ort::Value> inputTensor,output_tensors;
  inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memory_info, (float*)input_data.data(), input_data.size(), input_dims.data(), input_dims.size()));

  // run the inference
  bool ran_ok = ort_session_handler.infer(inputTensor, output_tensors, input_node_names, output_node_names);

  // we capture the model output dimensions
  Ort::TensorTypeAndShapeInfo outputInfo = output_tensors[0].GetTensorTypeAndShapeInfo();
  int batch_num = outputInfo.GetShape()[0];
  int channels = outputInfo.GetShape()[1];
  int height = outputInfo.GetShape()[2];
  int width = outputInfo.GetShape()[3];
  std::cout << batch_num << " " << channels << " " << height << " " << width << std::endl;
  
  // we want to read only the 8 masks coming from the network
  // this would need to be vector<vector<cv::mat>> for the higher batch numbers
  std::vector<cv::Mat> results;

  std::cout << "Reading batches now" << std::endl;

  for (int head_idx = 0; head_idx < batch_num; head_idx++)
  {
    float *output_data = output_tensors[head_idx].GetTensorMutableData<float>();
    results = ort_session_handler.postprocess(output_data, height, width, channels);
  }

  // now we want to classify each pixel using the individual masks
  cv::Mat segm = cv::Mat(original_height,original_width, CV_8U, cv::Scalar(0));

  // unclassified pixels are 0, everthing else gets a label value if semantic was picked up
  int label = 1;
  for(auto mat: results){
    cv::Mat temp = mat.clone();

    // set the zero as anything less than sigmoid value
    temp.setTo(0, temp < sigmoid_threshold);
    // set anything greater than sigmoid as the label for this segment
    temp.setTo(label, temp > sigmoid_threshold);
    // convert to unsigned char
    temp.convertTo(temp, CV_8U);
    // resize to the original image dimensions
    cv::resize(temp, temp, cv::Size(original_width, original_height), 0, 0, cv::INTER_CUBIC);
    // we add the classifications to each pixel
    segm += temp;
    // increment the label used
    label++;
  }

  // write the output
  cv::imwrite("segm.jpg",segm);

  exit(0);
}
