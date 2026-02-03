#include "llaisys/models/qwen2.h"
#include "../models/qwen2/qwen2_model.hpp"
#include<cstring>
#include "../utils.hpp"
#include<cmath>
#include <iostream>
struct LlaisysQwen2Model
{
    llaisys::models::Qwen2Model model;
};

__C struct LlaisysQwen2Model* llaisysQwen2ModelCreate(const LlaisysQwen2Meta* meta, llaisysDeviceType_t device, int* device_ids, int ndevice)
{
    if(meta == nullptr || device_ids == nullptr || ndevice <= 0) {
        return nullptr;
    }
    std::vector<int> dev_ids(device_ids, device_ids + ndevice);
    LlaisysQwen2Model* llaisys_model = new LlaisysQwen2Model{ llaisys::models::Qwen2Model(meta, device, dev_ids) };
    return llaisys_model;
}

__C void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model* model)
{
    if(model != nullptr) {
        delete model;
    }
}

__C struct LlaisysQwen2Weights* llaisysQwen2ModelWeights(struct LlaisysQwen2Model* model)
{
    if(model == nullptr) {
        return nullptr;
    }
    return model->model.weights();
}

__C int llaisysQwen2ModelInfer(struct LlaisysQwen2Model* model, int64_t* token_ids, size_t ntoken, float* probs_out, float temperature)
{
   if(model == nullptr || token_ids == nullptr || ntoken == 0 || probs_out == nullptr) {
       return -1; 
   }
   std::vector<int64_t> tokens(token_ids, token_ids + ntoken);
//    std::cout << "[C++ DEBUG] llaisysQwen2ModelInfer willbe called ntoken=" << ntoken << std::endl;
   llaisys::tensor_t logits = model->model.infer(tokens);
//    std::cout << "[C++ DEBUG] infer called, ntoken=" << ntoken << std::endl;
//   std::cout << "[C++ DEBUG] logits tensor=" << logits.get() << std::endl;
//   std::cout << "[C++ DEBUG] logits ndim=" << logits->ndim() << std::endl;
//   std::cout << "[C++ DEBUG] logits shape=[" << logits->shape()[0] << "," << logits->shape()[1] << "]" << std::endl;
   // Assuming logits is a 2D tensor with shape [1, vocab_size]
   // Copy logits data to probs_out
   size_t vocab_size = logits->shape()[1];
   std::vector<float> logits_f32(vocab_size);

  if (logits->dtype() == LLAISYS_DTYPE_F32) {
      // 直接拷贝
      std::memcpy(logits_f32.data(), logits->data(), vocab_size * sizeof(float));
  }
  else if (logits->dtype() == LLAISYS_DTYPE_BF16) {
      // BF16 -> F32
      const uint16_t* src = reinterpret_cast<const uint16_t*>(logits->data());
      for (size_t i = 0; i < vocab_size; i++) {
          llaisys::bf16_t val{src[i]};  
          logits_f32[i] = llaisys::utils::cast<float>(val);
      }
  }
  else if (logits->dtype() == LLAISYS_DTYPE_F16) {
      // F16 -> F32
      const uint16_t* src = reinterpret_cast<const uint16_t*>(logits->data());
      for (size_t i = 0; i < vocab_size; i++) {
          llaisys::fp16_t val{src[i]};
          logits_f32[i] = llaisys::utils::cast<float>(val);
      }
  }
  else {
      return -1;  
  }
      // 6. Temperature 
      if (temperature != 1.0f && temperature > 0.0f) {
          for (size_t i = 0; i < vocab_size; i++) {
              logits_f32[i] /= temperature;
          }
      }

      // 7. softmax
      float max_val = logits_f32[0];
      for (size_t i = 1; i < vocab_size; i++) {
          if (logits_f32[i] > max_val) max_val = logits_f32[i];
      }

      float sum = 0.0f;
      for (size_t i = 0; i < vocab_size; i++) {
          probs_out[i] = std::exp(logits_f32[i] - max_val);
          sum += probs_out[i];
      }

      for (size_t i = 0; i < vocab_size; i++) {
          probs_out[i] /= sum;
      }

      return 0;


}