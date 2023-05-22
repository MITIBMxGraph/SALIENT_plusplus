#include <iostream>
#include <torch/torch.h>

int main() {
    torch::Tensor n_id = torch::round(torch::randn({20})).to(torch::kLong);
    std::cout << n_id << std::endl;
}
