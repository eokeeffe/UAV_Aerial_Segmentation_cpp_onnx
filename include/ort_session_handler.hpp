#include <onnxruntime_cxx_api.h>

// if onnxruntime is built with cuda provider, the following header can be added to use cuda gpu
// #include <onnxruntime/core/providers/cuda/cuda_provider_factory.h>

#include <opencv2/opencv.hpp>

namespace deploy {
  class OrtSessionHandler {
   public:
    /**
     *  @param model_path path to onnx model
     */
    OrtSessionHandler(const std::string &model_path, 
      const std::vector<int64_t> &input_tensor_shapes, 
      bool use_cuda=false);

    virtual std::vector<float> preprocess(const cv::Mat &image, 
      int target_height, int target_width,
      const std::vector<float> &mean_val = {0.5, 0.5, 0.5},
      const std::vector<float> &std_val = {0.5, 0.5, 0.5}) const;

    bool infer(std::vector<Ort::Value>& input_tensors, std::vector<Ort::Value>& outputs, 
      std::vector<const char*> input_node_names, 
      std::vector<const char*> output_node_names);

    virtual std::vector<cv::Mat> postprocess(float *data, const int rows, const int cols, const int channels);

   private:
    std::string _model_path;
    std::vector<int64_t> _input_tensor_shapes;
    Ort::Env _env;
    std::unique_ptr<Ort::Session> _session;
  };
}  // namespace deploy