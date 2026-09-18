#include <algorithm>
#include <iterator>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include "include/json.hpp"

using tensor = std::vector<float>;
constexpr double EPSILON = 0.00001;

struct TransformerInput {
    // attention qkv w & b
    tensor q_weights, k_weights, v_weights;
    tensor q_biases, k_biases, v_biases;

    // mlp
    tensor l1_weights, l2_weights;
    tensor l1_biases, l2_biases;

    // layer norm
    tensor lnAttention_weights, lnAttenion_biases;
    tensor lnMlp_weights, lnMlp_biases;

    TransformerInput(
        tensor q_w, tensor k_w, tensor v_w,
        tensor q_b, tensor k_b, tensor v_b,
        tensor l1_w, tensor l2_w,
        tensor l1_b, tensor l2_b,
        tensor lnAtt_w, tensor lnAtt_b,
        tensor lnMlp_w, tensor lnMlp_b
    )
        : q_weights(q_w), k_weights(k_w), v_weights(v_w),
          q_biases(q_b), k_biases(k_b), v_biases(v_b),
          l1_weights(l1_w), l2_weights(l2_w),
          l1_biases(l1_b), l2_biases(l2_b),
          lnAttention_weights(lnAtt_w), lnAttenion_biases(lnAtt_b),
          lnMlp_weights(lnMlp_w), lnMlp_biases(lnMlp_b)
    {}
};

struct Gptweights {
    tensor embedding_weights, positionalEmbeddings;
    tensor final_bias, final_weights;
    std::vector<TransformerInput> transformer_weights;

    Gptweights(
        tensor emb_w, tensor pos_w,
        tensor fin_b, tensor fin_w,
        std::vector<TransformerInput> layers
    )
        : embedding_weights(emb_w), positionalEmbeddings(pos_w),
          final_bias(fin_b), final_weights(fin_w),
          transformer_weights(layers)
    {}
};

std::vector<std::string> gpt2tokens;
std::unordered_map<std::string, int> gpt2TokenToId;
std::map<std::pair<std::string, std::string>, int> merges;

std::string getTokenFromTokenId(int tokenId) {
    auto token = gpt2tokens[tokenId];
    return token;
}

tensor matMul(const tensor& a, const tensor& b, int n, int m, int p) {
    tensor result(n * p, 0.0f);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < m; k++) {
                result[i * p + j] += a[i * m + k] * b[k * p + j];
            }
        }
    }
    return result;
}

tensor transpose(const tensor& a, int n, int m) {
    tensor result(n * m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result[j * n + i] = a[i * m + j];
        }
    }
    return result;
}

float dotProduct(const tensor& a, const tensor& b) {
    float result = 0.0f;
    for (int i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

tensor addVectors(const tensor& a, const tensor& b) {
    tensor result(a.begin(), a.end());

    for (int i = 0; i < a.size(); i++) {
        result[i] += b[i];
    }
    return result;
}

tensor layerNorm(const tensor &embeddings, const tensor &weights, const tensor &biases){
    float mean = 0, variance = 0;
    int n = embeddings.size();

    for(auto &i : embeddings) {
        mean+=i;
    }
    mean /= n;

    for(auto &i : embeddings) {
        variance += (i-mean) * (i-mean);
    }
    variance /= n;

    float modifiedSD = sqrt(variance + EPSILON);

    tensor output(n);
    for(int i=0; i<n; i++){
        float normalizedVal = (embeddings[i] - mean) / modifiedSD;
        output[i] = normalizedVal * weights[i] + biases[i];
    }
    return output;
}

tensor softmax(const tensor &input){
    int n = input.size();
    float mx = *std::max_element(input.begin(), input.end());

    tensor result(n);
    float sum = 0.0f;
    for(int i = 0; i < n; i++){
        result[i] = exp(input[i] - mx);
        sum += result[i];
    }
    for(int i = 0; i < n; i++){
        result[i] /= sum;
    }
    return result;
}

float gelu(float x);

tensor forwardPass(const tensor &weights, const tensor &biases, const tensor &inputs, bool use_gelu = false) {
    int countNeurons = biases.size();
    tensor output(biases.begin(), biases.end());

    for(int i = 0; i < countNeurons; i++){
        for(int j = 0; j < inputs.size(); j++){
            output[i] += weights[i * inputs.size() + j] * inputs[j];
        }
    }

    if(use_gelu){
        for(int i = 0; i < countNeurons; i++){
            output[i] = gelu(output[i]);
        }
    }

    return output;
}

void parseMerges() {

}

void loadVocab() {
    using json = nlohmann::json;
    std::ifstream f("../weights/tokenizer/tokenizer.json");
    const json data = json::parse(f);
}

Gptweights loadWeights() {
    auto readFromFileStream = [](std::ifstream& stream, std::vector<float>& vec, int maxCount = 0) {
        float input;
        int count = 0;
        while (stream >> input) {
            vec.push_back(input);
            if(maxCount > 0 && ++count >= maxCount) break;
        }
    };

    const int N_EMBD = 768;
    const int N_LAYERS = 12;

    std::vector<TransformerInput> layers;

    for(int i = 0; i < N_LAYERS; i++) {
        std::string prefix = "../weights/transformer.h." + std::to_string(i) + ".";

        // layer norm 1
        tensor ln1_w, ln1_b;
        std::ifstream f_ln1w(prefix + "ln_1.weight.txt");
        std::ifstream f_ln1b(prefix + "ln_1.bias.txt");
        readFromFileStream(f_ln1w, ln1_w);
        readFromFileStream(f_ln1b, ln1_b);

        // fused qkv
        tensor qkv_w, qkv_b;
        std::ifstream f_qkvw(prefix + "attn.c_attn.weight.txt");
        std::ifstream f_qkvb(prefix + "attn.c_attn.bias.txt");
        readFromFileStream(f_qkvw, qkv_w);
        readFromFileStream(f_qkvb, qkv_b);

        int qkv_size = N_EMBD * N_EMBD;
        tensor q_w(qkv_w.begin(), qkv_w.begin() + qkv_size);
        tensor k_w(qkv_w.begin() + qkv_size, qkv_w.begin() + 2 * qkv_size);
        tensor v_w(qkv_w.begin() + 2 * qkv_size, qkv_w.end());

        tensor q_b(qkv_b.begin(), qkv_b.begin() + N_EMBD);
        tensor k_b(qkv_b.begin() + N_EMBD, qkv_b.begin() + 2 * N_EMBD);
        tensor v_b(qkv_b.begin() + 2 * N_EMBD, qkv_b.end());

        // attention projection
        tensor attn_proj_w, attn_proj_b;
        std::ifstream f_attnw(prefix + "attn.c_proj.weight.txt");
        std::ifstream f_attnb(prefix + "attn.c_proj.bias.txt");
        readFromFileStream(f_attnw, attn_proj_w);
        readFromFileStream(f_attnb, attn_proj_b);

        // layer norm 2
        tensor ln2_w, ln2_b;
        std::ifstream f_ln2w(prefix + "ln_2.weight.txt");
        std::ifstream f_ln2b(prefix + "ln_2.bias.txt");
        readFromFileStream(f_ln2w, ln2_w);
        readFromFileStream(f_ln2b, ln2_b);

        // mlp c_fc (up projection)
        tensor fc_w, fc_b;
        std::ifstream f_fcw(prefix + "mlp.c_fc.weight.txt");
        std::ifstream f_fcb(prefix + "mlp.c_fc.bias.txt");
        readFromFileStream(f_fcw, fc_w);
        readFromFileStream(f_fcb, fc_b);

        // mlp c_proj (down projection)
        tensor proj_w, proj_b;
        std::ifstream f_projwt(prefix + "mlp.c_proj.weight.txt");
        std::ifstream f_projb(prefix + "mlp.c_proj.bias.txt");
        readFromFileStream(f_projwt, proj_w);
        readFromFileStream(f_projb, proj_b);

        layers.push_back(TransformerInput(
            q_w, k_w, v_w, q_b, k_b, v_b,
            fc_w, proj_w, fc_b, proj_b,
            ln1_w, ln1_b, ln2_w, ln2_b
        ));
    }

    // token and positional embeddings
    tensor tok_w, pos_w;
    std::ifstream f_tokw("../weights/transformer.wte.weight.txt");
    std::ifstream f_posw("../weights/transformer.wpe.weight.txt");
    readFromFileStream(f_tokw, tok_w);
    readFromFileStream(f_posw, pos_w);

    // final layer norm
    tensor lnf_w, lnf_b;
    std::ifstream f_lnfw("../weights/transformer.ln_f.weight.txt");
    std::ifstream f_lnfb("../weights/transformer.ln_f.bias.txt");
    readFromFileStream(f_lnfw, lnf_w);
    readFromFileStream(f_lnfb, lnf_b);

    return Gptweights(tok_w, pos_w, lnf_b, lnf_w, layers);
}

float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

void getTokenEmbedding() {

}

void gelu() {

}

int main() {
    // load weights
    Gptweights weights = loadWeights();

    // 
}