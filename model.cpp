#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, const char* argv[]) {
    // Check command-line arguments
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_script_model>\n";
        return -1;
    }

    // Load the TorchScript model
    //std::string model_path = argv[1];
    std::string model_path = "vit_cifar10_model_script.pt";
    torch::jit::Module module = torch::jit::load(model_path);
    //std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(model_path);

    // Load the CIFAR-10 test set and perform inference
    // You need to implement the data loading and inference part in C++ based on your setup.

    // Perform inference on test data
    // Replace this part with your actual inference code in C++
    // For example:
    torch::Tensor input_data = torch::randn({1, 3, 224, 224});  // Example input data
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_data);
    at::Tensor output = module.forward(inputs).toTensor();
    //at::Tensor output = module->forward(inputs).toTensor();
    
    // Print the predicted class
    auto predicted_class = output.argmax(1).item<int>();
    std::cout << "Predicted class: " << predicted_class << std::endl;

    return 0;
}

