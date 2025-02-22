#include "ort_session_handler.hpp"

namespace deploy {
OrtSessionHandler::OrtSessionHandler(const std::string &model_path,
                                     const std::vector<int64_t> &input_tensor_shapes, bool use_cuda)
    : _model_path(model_path),
      _input_tensor_shapes(input_tensor_shapes),
      _env(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ort session handler"))
{

  Ort::SessionOptions session_options;

  session_options.SetInterOpNumThreads(1);
  session_options.SetIntraOpNumThreads(1);
  // Optimization will take time and memory during startup
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

  if(use_cuda){
    OrtCUDAProviderOptions cuda_options;
    // if onnxruntime is built with cuda provider, the following function can be added to use cuda gpu
    cuda_options.device_id = 0;  //GPU_ID
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive; // Algo to search for Cudnn
    cuda_options.arena_extend_strategy = 0;
    // May cause data race in some condition
    cuda_options.do_copy_in_default_stream = 0;
    session_options.AppendExecutionProvider_CUDA(cuda_options);
  }

  _session.reset( new Ort::Session(_env, _model_path.c_str(), session_options) );
}

std::vector<float> OrtSessionHandler::preprocess(const cv::Mat &image, int target_height, int target_width,
                                                 const std::vector<float> &mean_val,
                                                 const std::vector<float> &std_val) const {
  if (image.empty() || image.channels() != 3) {
    throw std::runtime_error("invalid image");
  }

  if (target_height * target_width == 0) {
    throw std::runtime_error("invalid target dimension");
  }

  cv::Mat processed = image.clone();

  if (image.rows != target_height || image.cols != target_width) {
    cv::resize(processed, processed, cv::Size(target_width, target_height), 0, 0, cv::INTER_CUBIC);
  }
  cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
  std::vector<float> data(3 * target_height * target_width);

  for (int i = 0; i < target_height; ++i) {
    for (int j = 0; j < target_width; ++j) {
      for (int c = 0; c < 3; ++c) {
        data[c * target_height * target_width + i * target_width + j] =
            (processed.data[i * target_width * 3 + j * 3 + c] / 255.0 - mean_val[c]) / std_val[c];
      }
    }
  }

  return data;
}

bool OrtSessionHandler::infer(std::vector<Ort::Value>& input_tensors, std::vector<Ort::Value>& outputs, 
  std::vector<const char*> input_node_names, std::vector<const char*> output_node_names)
{
  try {
    outputs = _session->Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), input_tensors.size(), output_node_names.data(), 1);
  }
  catch (Ort::Exception oe) {
    std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
    return false;
  }

  return true;
}

std::vector<cv::Mat> OrtSessionHandler::postprocess(float *data, const int rows, const int cols, const int channels)
{
    // data is in chw format, we need hwc format for OpenCV cv::Mat container

    // Create stacked vector of cv::Mats
    std::vector<cv::Mat> stacked_mats;
    stacked_mats.reserve(channels);  // Reserve space for efficiency

    for (int c = 0; c < channels; ++c) {
      cv::Mat channel(rows, cols, CV_32F, cv::Scalar(0.0));

      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          channel.at<float>(i,j) = data[c * rows * cols + i * cols + j];
        }
      }
      stacked_mats.push_back(channel);
    }
    return stacked_mats;
}

}  // namespace deploy