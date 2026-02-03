#include "qwen2_model.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/add/op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../llaisys/llaisys_tensor.hpp"
#include "../../utils.hpp"
#include <cmath>
#include <cstring>
#include <iostream>

namespace llaisys {
namespace models {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta* meta, 
                       llaisysDeviceType_t device_type, 
                       const std::vector<int>& device_ids) {
    // std::cerr << "[C++ DEBUG] Qwen2Model constructor started" << std::endl;
    config_.nlayer = meta->nlayer;
    config_.hidden_size = meta->hs;
    config_.num_query_heads = meta->nh;
    config_.num_kv_heads = meta->nkvh;
    config_.head_dim = meta->dh;
    config_.intermediate_size = meta->di;
    config_.max_seq_Len = meta->maxseq;
    config_.vocab_size = meta->voc;
    config_.rms_norm_eps = meta->epsilon;
    config_.rope_theta = meta->theta;
    config_.eos_token_id = meta->end_token;
    config_.dtype = meta->dtype;
    config_.device = device_type;
    cache_len_ = 0;

    // std::cerr << "[C++ DEBUG] Creating weights_, nlayer=" << config_.nlayer << std::endl;
    // 初始化 weights_
    weights_ = new LlaisysQwen2Weights();
    // std::cerr << "[C++ DEBUG] weights_ created at " << weights_ << std::endl;

    size_t in_embed_shape[2] = {config_.vocab_size, config_.hidden_size};
    size_t out_embed_shape[2] = {config_.vocab_size, config_.hidden_size};
    size_t out_norm_w_shape[1] = {config_.hidden_size};

    size_t atten_norm_w_shape[1] = {config_.hidden_size};
    size_t atten_q_w_shape[2] = {config_.num_query_heads * config_.head_dim, config_.hidden_size};
    size_t atten_q_b_shape[1] = {config_.num_query_heads * config_.head_dim};
    size_t atten_k_w_shape[2] = {config_.num_kv_heads * config_.head_dim, config_.hidden_size};
    size_t atten_k_b_shape[1] = {config_.num_kv_heads * config_.head_dim};
    size_t atten_v_w_shape[2] = {config_.num_kv_heads * config_.head_dim, config_.hidden_size};
    size_t atten_v_b_shape[1] = {config_.num_kv_heads * config_.head_dim};
    size_t atten_o_w_shape[2] = {config_.hidden_size, config_.hidden_size};
    size_t mlp_norm_w_shape[1] = {config_.hidden_size};
    size_t mlp_gate_w_shape[2] = {config_.intermediate_size, config_.hidden_size};
    size_t mlp_up_w_shape[2] = {config_.intermediate_size, config_.hidden_size};
    size_t mlp_down_w_shape[2] = {config_.hidden_size, config_.intermediate_size};

    // std::cerr << "[C++ DEBUG] Creating in_embed tensor..." << std::endl;
    weights_->in_embed = tensorCreate(in_embed_shape, 2, config_.dtype, config_.device, device_ids[0]);
    // std::cerr << "[C++ DEBUG] in_embed=" << weights_->in_embed << std::endl;
    
    // std::cerr << "[C++ DEBUG] Creating out_embed tensor..." << std::endl;
    weights_->out_embed = tensorCreate(out_embed_shape, 2, config_.dtype, config_.device, device_ids[0]);
    // std::cerr << "[C++ DEBUG] out_embed=" << weights_->out_embed << std::endl;
    
    // std::cerr << "[C++ DEBUG] Creating out_norm_w tensor..." << std::endl;
    weights_->out_norm_w = tensorCreate(out_norm_w_shape, 1, config_.dtype, config_.device, device_ids[0]);
    // std::cerr << "[C++ DEBUG] out_norm_w=" << weights_->out_norm_w << std::endl;
    
    // std::cerr << "[C++ DEBUG] Allocating weight arrays for " << config_.nlayer << " layers..." << std::endl;
    weights_->attn_norm_w = new llaisysTensor_t[config_.nlayer];
    weights_->attn_q_w = new llaisysTensor_t[config_.nlayer];
    weights_->attn_q_b = new llaisysTensor_t[config_.nlayer];
    weights_->attn_k_w = new llaisysTensor_t[config_.nlayer];
    weights_->attn_k_b = new llaisysTensor_t[config_.nlayer];
    weights_->attn_v_w = new llaisysTensor_t[config_.nlayer];
    weights_->attn_v_b = new llaisysTensor_t[config_.nlayer];
    weights_->attn_o_w = new llaisysTensor_t[config_.nlayer];
    weights_->mlp_norm_w = new llaisysTensor_t[config_.nlayer];
    weights_->mlp_gate_w = new llaisysTensor_t[config_.nlayer];
    weights_->mlp_up_w = new llaisysTensor_t[config_.nlayer];
    weights_->mlp_down_w = new llaisysTensor_t[config_.nlayer];
    // std::cerr << "[C++ DEBUG] Weight arrays allocated" << std::endl;
    
    // std::cerr << "[C++ DEBUG] Creating layer weights..." << std::endl;
    for (size_t i = 0; i < config_.nlayer; i++) {
        weights_->attn_norm_w[i] = tensorCreate(atten_norm_w_shape, 1, config_.dtype, config_.device, device_ids[0]);
        weights_->attn_q_w[i] = tensorCreate(atten_q_w_shape, 2, config_.dtype, config_.device, device_ids[0]);
        weights_->attn_q_b[i] = tensorCreate(atten_q_b_shape, 1, config_.dtype, config_.device, device_ids[0]);
        weights_->attn_k_w[i] = tensorCreate(atten_k_w_shape, 2, config_.dtype, config_.device, device_ids[0]);
        weights_->attn_k_b[i] = tensorCreate(atten_k_b_shape, 1, config_.dtype, config_.device, device_ids[0]);
        weights_->attn_v_w[i] = tensorCreate(atten_v_w_shape, 2, config_.dtype, config_.device, device_ids[0]);
        weights_->attn_v_b[i] = tensorCreate(atten_v_b_shape, 1, config_.dtype, config_.device, device_ids[0]);
        weights_->attn_o_w[i] = tensorCreate(atten_o_w_shape, 2, config_.dtype, config_.device, device_ids[0]);
        weights_->mlp_norm_w[i] = tensorCreate(mlp_norm_w_shape, 1, config_.dtype, config_.device, device_ids[0]);
        weights_->mlp_gate_w[i] = tensorCreate(mlp_gate_w_shape, 2, config_.dtype, config_.device, device_ids[0]);
        weights_->mlp_up_w[i] = tensorCreate(mlp_up_w_shape, 2, config_.dtype, config_.device, device_ids[0]);
        weights_->mlp_down_w[i] = tensorCreate(mlp_down_w_shape, 2, config_.dtype, config_.device, device_ids[0]);
        if (i == 0) {
            // std::cerr << "[C++ DEBUG] Layer 0 weights created, attn_q_w[0]=" << weights_->attn_q_w[0] << std::endl;
        }
    }
    // std::cerr << "[C++ DEBUG] All layer weights created" << std::endl;
    
    // 初始化 KV Cache，预先分配最大长度
    // std::cerr << "[C++ DEBUG] Creating KV Cache, max_seq_Len=" << config_.max_seq_Len 
    //           << " num_kv_heads=" << config_.num_kv_heads 
    //           << " head_dim=" << config_.head_dim << std::endl;
    for (size_t layer = 0; layer < config_.nlayer; layer++) {
        llaisys::tensor_t k_cache_tensor = llaisys::Tensor::create(
            {config_.max_seq_Len, config_.num_kv_heads, config_.head_dim}, 
            config_.dtype, config_.device, 0);
        llaisys::tensor_t v_cache_tensor = llaisys::Tensor::create(
            {config_.max_seq_Len, config_.num_kv_heads, config_.head_dim}, 
            config_.dtype, config_.device, 0);
        if (!k_cache_tensor || !v_cache_tensor) {
            // std::cerr << "[C++ ERROR] Failed to create KV cache for layer " << layer << std::endl;
        }
        kv_cache_.k_cache.push_back(k_cache_tensor);
        kv_cache_.v_cache.push_back(v_cache_tensor);
        if (layer == 0) {
            // std::cerr << "[C++ DEBUG] KV cache layer 0 created" << std::endl;
        }
    }
    // std::cerr << "[C++ DEBUG] Qwen2Model constructor completed" << std::endl;
}

Qwen2Model::~Qwen2Model() {
    tensorDestroy(weights_->in_embed);
    tensorDestroy(weights_->out_embed);
    tensorDestroy(weights_->out_norm_w);
    
    for (size_t i = 0; i < config_.nlayer; i++) {
        tensorDestroy(weights_->attn_norm_w[i]);
        tensorDestroy(weights_->attn_q_w[i]);
        tensorDestroy(weights_->attn_q_b[i]);
        tensorDestroy(weights_->attn_k_w[i]);
        tensorDestroy(weights_->attn_k_b[i]);
        tensorDestroy(weights_->attn_v_w[i]);
        tensorDestroy(weights_->attn_v_b[i]);
        tensorDestroy(weights_->attn_o_w[i]);
        tensorDestroy(weights_->mlp_norm_w[i]);
        tensorDestroy(weights_->mlp_gate_w[i]);
        tensorDestroy(weights_->mlp_up_w[i]);
        tensorDestroy(weights_->mlp_down_w[i]);
    }
    
    delete[] weights_->attn_norm_w;
    delete[] weights_->attn_q_w;
    delete[] weights_->attn_q_b;
    delete[] weights_->attn_k_w;
    delete[] weights_->attn_k_b;
    delete[] weights_->attn_v_w;
    delete[] weights_->attn_v_b;
    delete[] weights_->attn_o_w;
    delete[] weights_->mlp_norm_w;
    delete[] weights_->mlp_gate_w;
    delete[] weights_->mlp_up_w;
    delete[] weights_->mlp_down_w;
    
    delete weights_;
    
};

llaisys::tensor_t Qwen2Model::infer(const std::vector<int64_t>& token_ids) {
//     std::cerr << "[DEBUG infer] Start, token_ids.size=" << token_ids.size() << std::endl;
    
    // 检查 weights_ 是否为空
    if (!weights_) {
//         std::cerr << "[ERROR] weights_ is null!" << std::endl;
        return nullptr;
    }
    
    // 检查关键权重是否为空
    if (!weights_->in_embed) {
//         std::cerr << "[ERROR] weights_->in_embed is null!" << std::endl;
        return nullptr;
    }
//     std::cerr << "[DEBUG infer] weights_->in_embed is valid" << std::endl;
    
    int seq_len = token_ids.size();
    int total_len = cache_len_ + seq_len;
//     std::cerr << "[DEBUG infer] seq_len=" << seq_len << ", cache_len_=" << cache_len_ << ", total_len=" << total_len << std::endl;
    
    // 输入的 token_ids 先进行 embedding
//     std::cerr << "[DEBUG infer] Creating input_ids_tensor..." << std::endl;
    llaisys::tensor_t input_ids_tensor = llaisys::Tensor::create(
        {(size_t)seq_len}, LLAISYS_DTYPE_I64, config_.device, 0);
//     std::cerr << "[DEBUG infer] Loading input_ids..." << std::endl;
    input_ids_tensor->load(token_ids.data());
    
//     std::cerr << "[DEBUG infer] Creating embedded tensor..." << std::endl;
    llaisys::tensor_t embedded = llaisys::Tensor::create(
        {(size_t)seq_len, config_.hidden_size}, config_.dtype, config_.device, 0);
//     std::cerr << "[DEBUG infer] About to call embedding" << std::endl;
    if (!weights_->in_embed->tensor) {
//         std::cerr << "[ERROR] weights_->in_embed->tensor is null!" << std::endl;
        return nullptr;
    }
    llaisys::ops::embedding(embedded, input_ids_tensor, weights_->in_embed->tensor);
//     std::cerr << "[DEBUG infer] === EMBEDDING DONE ===" << std::endl;
    
    llaisys::tensor_t x = embedded;
//     std::cerr << "[DEBUG infer] x tensor shape=[" << x->shape()[0] << "," << x->shape()[1] << "]" << std::endl;
    
    // 准备位置编码（只在循环外创建一次）
//     std::cerr << "[DEBUG infer] Preparing pos_ids..." << std::endl;
    std::vector<int64_t> pos_ids(seq_len);
    for (int i = 0; i < seq_len; i++) {
        pos_ids[i] = cache_len_ + i;
    }
//     std::cerr << "[DEBUG infer] Creating pos_ids_tensor..." << std::endl;
    llaisys::tensor_t pos_ids_tensor = llaisys::Tensor::create(
        {(size_t)seq_len}, LLAISYS_DTYPE_I64, config_.device, 0);
    pos_ids_tensor->load(pos_ids.data());
//     std::cerr << "[DEBUG infer] pos_ids_tensor created" << std::endl;
    
//     std::cerr << "[DEBUG infer] Starting layer loop, nlayer=" << config_.nlayer << std::endl;
    for (size_t layer = 0; layer < config_.nlayer; layer++) {
//         std::cerr << "[DEBUG infer] ===== Layer " << layer << " =====" << std::endl;
        
        // 1. RMSNorm (attention 之前的)
//         std::cerr << "[DEBUG infer] Layer " << layer << " Step 1: RMSNorm..." << std::endl;
        llaisys::tensor_t x_norm = llaisys::Tensor::create(
            x->shape(), x->dtype(), x->deviceType(), x->deviceId());
//         std::cerr << "[DEBUG infer] Layer " << layer << " x_norm created" << std::endl;
        
        // 检查权重
        if (!weights_->attn_norm_w[layer]) {
//             std::cerr << "[ERROR] Layer " << layer << " weights_->attn_norm_w[layer] is null!" << std::endl;
            return nullptr;
        }
        if (!weights_->attn_norm_w[layer]->tensor) {
//             std::cerr << "[ERROR] Layer " << layer << " weights_->attn_norm_w[layer]->tensor is null!" << std::endl;
            return nullptr;
        }
//         std::cerr << "[DEBUG infer] Layer " << layer << " Calling rms_norm..." << std::endl;
        llaisys::ops::rms_norm(x_norm, x, weights_->attn_norm_w[layer]->tensor, config_.rms_norm_eps);
//         std::cerr << "[DEBUG infer] Layer " << layer << " RMSNorm done" << std::endl;

        // 2. Q, K, V 投影
//         std::cerr << "[DEBUG infer] Layer " << layer << " Step 2: QKV Linear..." << std::endl;
        llaisys::tensor_t q_proj = llaisys::Tensor::create(
            {(size_t)seq_len, config_.num_query_heads * config_.head_dim}, 
            x->dtype(), x->deviceType(), x->deviceId());
        llaisys::tensor_t k_proj = llaisys::Tensor::create(
            {(size_t)seq_len, config_.num_kv_heads * config_.head_dim}, 
            x->dtype(), x->deviceType(), x->deviceId());
        llaisys::tensor_t v_proj = llaisys::Tensor::create(
            {(size_t)seq_len, config_.num_kv_heads * config_.head_dim}, 
            x->dtype(), x->deviceType(), x->deviceId());
//         std::cerr << "[DEBUG infer] Layer " << layer << " QKV tensors created" << std::endl;

        // 检查权重
        if (!weights_->attn_q_w[layer] || !weights_->attn_q_b[layer]) {
//             std::cerr << "[ERROR] Layer " << layer << " Q weights are null!" << std::endl;
            return nullptr;
        }
        if (!weights_->attn_k_w[layer] || !weights_->attn_k_b[layer]) {
//             std::cerr << "[ERROR] Layer " << layer << " K weights are null!" << std::endl;
            return nullptr;
        }
        if (!weights_->attn_v_w[layer] || !weights_->attn_v_b[layer]) {
//             std::cerr << "[ERROR] Layer " << layer << " V weights are null!" << std::endl;
            return nullptr;
        }
        
//         std::cerr << "[DEBUG infer] Layer " << layer << " Calling linear Q..." << std::endl;
        llaisys::ops::linear(q_proj, x_norm, weights_->attn_q_w[layer]->tensor, weights_->attn_q_b[layer]->tensor);
//         std::cerr << "[DEBUG infer] Layer " << layer << " Calling linear K..." << std::endl;
        llaisys::ops::linear(k_proj, x_norm, weights_->attn_k_w[layer]->tensor, weights_->attn_k_b[layer]->tensor);
//         std::cerr << "[DEBUG infer] Layer " << layer << " Calling linear V..." << std::endl;
        llaisys::ops::linear(v_proj, x_norm, weights_->attn_v_w[layer]->tensor, weights_->attn_v_b[layer]->tensor);
//         std::cerr << "[DEBUG infer] Layer " << layer << " QKV Linear done" << std::endl;

        // view 为 [seq_len, num_heads, head_dim]
//         std::cerr << "[DEBUG infer] Layer " << layer << " Step 3: View QKV..." << std::endl;
        llaisys::tensor_t q = q_proj->view({(size_t)seq_len, config_.num_query_heads, config_.head_dim});
        llaisys::tensor_t k = k_proj->view({(size_t)seq_len, config_.num_kv_heads, config_.head_dim});
        llaisys::tensor_t v = v_proj->view({(size_t)seq_len, config_.num_kv_heads, config_.head_dim});
//         std::cerr << "[DEBUG infer] Layer " << layer << " View done" << std::endl;

        // 3. 对 Q 和 K 应用 RoPE 位置编码
//         std::cerr << "[DEBUG infer] Layer " << layer << " Step 4: RoPE..." << std::endl;
        llaisys::tensor_t q_rope = llaisys::Tensor::create(
            q->shape(), q->dtype(), q->deviceType(), q->deviceId());
        llaisys::tensor_t k_rope = llaisys::Tensor::create(
            k->shape(), k->dtype(), k->deviceType(), k->deviceId());
//         std::cerr << "[DEBUG infer] Layer " << layer << " Calling rope Q..." << std::endl;
        llaisys::ops::rope(q_rope, q, pos_ids_tensor, config_.rope_theta);
//         std::cerr << "[DEBUG infer] Layer " << layer << " Calling rope K..." << std::endl;
        llaisys::ops::rope(k_rope, k, pos_ids_tensor, config_.rope_theta);
//         std::cerr << "[DEBUG infer] Layer " << layer << " RoPE done" << std::endl;

        // 4. 将应用 RoPE 后的 K 和原始 V 写入 KV Cache
//         std::cerr << "[DEBUG infer] Layer " << layer << " Step 5: KV Cache write..." << std::endl;
//         std::cerr << "[DEBUG infer] Layer " << layer << " cache_len_=" << cache_len_ << ", total_len=" << total_len << std::endl;
        llaisys::tensor_t k_cache_slice = kv_cache_.k_cache[layer]->slice(0, cache_len_, total_len);
        llaisys::tensor_t v_cache_slice = kv_cache_.v_cache[layer]->slice(0, cache_len_, total_len);
//         std::cerr << "[DEBUG infer] Layer " << layer << " KV Cache slices created" << std::endl;
        
        size_t kv_bytes = seq_len * config_.num_kv_heads * config_.head_dim * utils::dsize(config_.dtype);
//         std::cerr << "[DEBUG infer] Layer " << layer << " kv_bytes=" << kv_bytes << std::endl;
        
        void* k_rope_data = k_rope->data();
        void* v_data = v->data();
        void* k_cache_data = k_cache_slice->data();
        void* v_cache_data = v_cache_slice->data();
        
//         std::cerr << "[DEBUG infer] Layer " << layer << " k_rope_data=" << k_rope_data << std::endl;
//         std::cerr << "[DEBUG infer] Layer " << layer << " v_data=" << v_data << std::endl;
//         std::cerr << "[DEBUG infer] Layer " << layer << " k_cache_data=" << k_cache_data << std::endl;
//         std::cerr << "[DEBUG infer] Layer " << layer << " v_cache_data=" << v_cache_data << std::endl;
        
        if (!k_rope_data || !v_data || !k_cache_data || !v_cache_data) {
//             std::cerr << "[ERROR] Layer " << layer << " Some tensor data is null!" << std::endl;
            return nullptr;
        }
        
//         std::cerr << "[DEBUG infer] Layer " << layer << " memcpy k_rope..." << std::endl;
        std::memcpy(k_cache_data, k_rope_data, kv_bytes);
//         std::cerr << "[DEBUG infer] Layer " << layer << " memcpy v..." << std::endl;
        std::memcpy(v_cache_data, v_data, kv_bytes);
//         std::cerr << "[DEBUG infer] Layer " << layer << " KV Cache write done" << std::endl;

        // 5. 读取全部历史 K, V（K 已经应用过 RoPE）
//         std::cerr << "[DEBUG infer] Layer " << layer << " Step 6: KV Cache read..." << std::endl;
        llaisys::tensor_t k_full = kv_cache_.k_cache[layer]->slice(0, 0, total_len);
        llaisys::tensor_t v_full = kv_cache_.v_cache[layer]->slice(0, 0, total_len);
//         std::cerr << "[DEBUG infer] Layer " << layer << " k_full shape=[" << k_full->shape()[0] << "," << k_full->shape()[1] << "," << k_full->shape()[2] << "]" << std::endl;
//         std::cerr << "[DEBUG infer] Layer " << layer << " v_full shape=[" << v_full->shape()[0] << "," << v_full->shape()[1] << "," << v_full->shape()[2] << "]" << std::endl;
//         std::cerr << "[DEBUG infer] Layer " << layer << " KV Cache read done" << std::endl;

        // 6. Self Attention (GQA)
//         std::cerr << "[DEBUG infer] Layer " << layer << " Step 7: Self Attention..." << std::endl;
        llaisys::tensor_t attn_out = llaisys::Tensor::create(
            {(size_t)seq_len, config_.num_query_heads, config_.head_dim}, 
            x->dtype(), x->deviceType(), x->deviceId());
        float scale = 1.0f / sqrtf(static_cast<float>(config_.head_dim));
//         std::cerr << "[DEBUG infer] Layer " << layer << " scale=" << scale << std::endl;
//         std::cerr << "[DEBUG infer] Layer " << layer << " Calling self_attention..." << std::endl;
//         std::cerr << "[DEBUG infer] Layer " << layer << " q_rope shape=[" << q_rope->shape()[0] << "," << q_rope->shape()[1] << "," << q_rope->shape()[2] << "]" << std::endl;
//         std::cerr << "[DEBUG infer] Layer " << layer << " k_full shape=[" << k_full->shape()[0] << "," << k_full->shape()[1] << "," << k_full->shape()[2] << "]" << std::endl;
//         std::cerr << "[DEBUG infer] Layer " << layer << " v_full shape=[" << v_full->shape()[0] << "," << v_full->shape()[1] << "," << v_full->shape()[2] << "]" << std::endl;
        llaisys::ops::self_attention(attn_out, q_rope, k_full, v_full, scale);
//         std::cerr << "[DEBUG infer] Layer " << layer << " Self Attention done" << std::endl;

        // 7. Output projection + residual
//         std::cerr << "[DEBUG infer] Layer " << layer << " Step 8: Output projection..." << std::endl;
        llaisys::tensor_t attn_out_2d = attn_out->view({(size_t)seq_len, config_.hidden_size});
//         std::cerr << "[DEBUG infer] Layer " << layer << " attn_out_2d shape=[" << attn_out_2d->shape()[0] << "," << attn_out_2d->shape()[1] << "]" << std::endl;
        llaisys::tensor_t o_out = llaisys::Tensor::create(
            {(size_t)seq_len, config_.hidden_size}, 
            x->dtype(), x->deviceType(), x->deviceId());
//         std::cerr << "[DEBUG infer] Layer " << layer << " Calling linear output projection..." << std::endl;
        llaisys::ops::linear(o_out, attn_out_2d, weights_->attn_o_w[layer]->tensor, nullptr);
//         std::cerr << "[DEBUG infer] Layer " << layer << " Output projection done" << std::endl;
        
        // residual connection
//         std::cerr << "[DEBUG infer] Layer " << layer << " Step 9: First residual..." << std::endl;
        llaisys::tensor_t after_attn = llaisys::Tensor::create(
            x->shape(), x->dtype(), x->deviceType(), x->deviceId());
        llaisys::ops::add(after_attn, x, o_out);
//         std::cerr << "[DEBUG infer] Layer " << layer << " First residual done" << std::endl;

        // 8. RMSNorm + MLP
//         std::cerr << "[DEBUG infer] Layer " << layer << " Step 10: MLP RMSNorm..." << std::endl;
        llaisys::tensor_t x_mlp_norm = llaisys::Tensor::create(
            after_attn->shape(), after_attn->dtype(), after_attn->deviceType(), after_attn->deviceId());
        llaisys::ops::rms_norm(x_mlp_norm, after_attn, weights_->mlp_norm_w[layer]->tensor, config_.rms_norm_eps);
//         std::cerr << "[DEBUG infer] Layer " << layer << " MLP RMSNorm done" << std::endl;
        
        // MLP
//         std::cerr << "[DEBUG infer] Layer " << layer << " Step 11: MLP Linear..." << std::endl;
        llaisys::tensor_t gate = llaisys::Tensor::create(
            {(size_t)seq_len, config_.intermediate_size}, 
            x->dtype(), x->deviceType(), x->deviceId());
        llaisys::tensor_t up = llaisys::Tensor::create(
            {(size_t)seq_len, config_.intermediate_size}, 
            x->dtype(), x->deviceType(), x->deviceId());
        
        llaisys::ops::linear(gate, x_mlp_norm, weights_->mlp_gate_w[layer]->tensor, nullptr);
        llaisys::ops::linear(up, x_mlp_norm, weights_->mlp_up_w[layer]->tensor, nullptr);
//         std::cerr << "[DEBUG infer] Layer " << layer << " MLP Linear done" << std::endl;

        // SwiGLU
//         std::cerr << "[DEBUG infer] Layer " << layer << " Step 12: SwiGLU..." << std::endl;
        llaisys::tensor_t swiglu_out = llaisys::Tensor::create(
            {(size_t)seq_len, config_.intermediate_size}, 
            x->dtype(), x->deviceType(), x->deviceId());
        llaisys::ops::swiglu(swiglu_out, gate, up);
//         std::cerr << "[DEBUG infer] Layer " << layer << " SwiGLU done" << std::endl;

        // MLP output projection
//         std::cerr << "[DEBUG infer] Layer " << layer << " Step 13: MLP output projection..." << std::endl;
        llaisys::tensor_t mlp_out = llaisys::Tensor::create(
            {(size_t)seq_len, config_.hidden_size}, 
            x->dtype(), x->deviceType(), x->deviceId());
        llaisys::ops::linear(mlp_out, swiglu_out, weights_->mlp_down_w[layer]->tensor, nullptr);
//         std::cerr << "[DEBUG infer] Layer " << layer << " MLP output projection done" << std::endl;

        // residual connection
//         std::cerr << "[DEBUG infer] Layer " << layer << " Step 14: Second residual..." << std::endl;
        llaisys::tensor_t final_out = llaisys::Tensor::create(
            x->shape(), x->dtype(), x->deviceType(), x->deviceId());
        llaisys::ops::add(final_out, after_attn, mlp_out);
        x = final_out;
//         std::cerr << "[DEBUG infer] Layer " << layer << " ===== DONE =====" << std::endl;
    }
    
    // 最终的 RMSNorm
//     std::cerr << "[DEBUG infer] Final RMSNorm..." << std::endl;
    llaisys::tensor_t x_final_norm = llaisys::Tensor::create(
        x->shape(), x->dtype(), x->deviceType(), x->deviceId());
    llaisys::ops::rms_norm(x_final_norm, x, weights_->out_norm_w->tensor, config_.rms_norm_eps);
//     std::cerr << "[DEBUG infer] Final RMSNorm done" << std::endl;
    
    // LM Head (取最后一个位置)
//     std::cerr << "[DEBUG infer] LM Head slice..." << std::endl;
    llaisys::tensor_t last_hidden = x_final_norm->slice(0, seq_len - 1, seq_len); // [1, hidden_size]
//     std::cerr << "[DEBUG infer] last_hidden shape=[" << last_hidden->shape()[0] << "," << last_hidden->shape()[1] << "]" << std::endl;
    
//     std::cerr << "[DEBUG infer] Creating logits tensor..." << std::endl;
    llaisys::tensor_t logits = llaisys::Tensor::create(
        {1, config_.vocab_size}, x->dtype(), x->deviceType(), x->deviceId());
//     std::cerr << "[DEBUG infer] Calling final linear..." << std::endl;
    llaisys::ops::linear(logits, last_hidden, weights_->out_embed->tensor, nullptr);
//     std::cerr << "[DEBUG infer] Final linear done" << std::endl;
    
    // 更新 cache 长度
    cache_len_ += seq_len;
    
//     std::cerr << "[DEBUG infer] === INFER DONE ===" << std::endl;
    return logits;
}

void Qwen2Model::reset_cache() {
    cache_len_ = 0;
}

} // namespace models
} // namespace llaisys
