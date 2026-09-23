#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "include/json.hpp"
#include<cmath>

using tensor = std::vector<float>;

std::string weightsDir = "../weights/qwen/";

struct Config {
    int hiddenSize, numLayers, numHeads, numKVHeads, intermediateSize, vocabSize, maxPos;
    float ropeTheta, rmsNormEps;
    bool tieWordEmbeddings;

    int headDim() const;
    int headsPerKV() const;
};

Config loadConfig() {
    using json = nlohmann::json;
    std::ifstream f(weightsDir + "config.json");
    if (!f) { 
        std::cerr << "missing config.json\n"; std::exit(1);
    }
    const json j = json::parse(f);

    Config cfg;
    // read by direct key extraction using json object j
    cfg.hiddenSize       = j["hidden_size"];
    cfg.numLayers        = j["num_hidden_layers"];
    cfg.numHeads         = j["num_attention_heads"];
    cfg.numKVHeads       = j["num_key_value_heads"];
    cfg.intermediateSize = j["intermediate_size"];
    cfg.vocabSize        = j["vocab_size"];
    cfg.maxPos           = j["max_position_embeddings"];
    cfg.ropeTheta        = j["rope_theta"];
    cfg.rmsNormEps       = j["rms_norm_eps"];
    cfg.tieWordEmbeddings = j["tie_word_embeddings"];
    return cfg;
}

int Config::headDim() const { return hiddenSize / numHeads; }
int Config::headsPerKV() const { return numHeads / numKVHeads; }

// 8 byte header length, then a json index of every tensor's dtype/shape/byte range,
// then the raw bytes. qwen ships bf16.
struct Safetensors {
    struct Entry { std::string dtype; std::vector<int> shape; size_t begin, end; };
    std::string blob;
    size_t dataStart;
    std::unordered_map<std::string, Entry> index;

    void load(const std::string &path);
    const Entry& at(const std::string &name) const;
    tensor get(const std::string &name) const;
};

struct QwenLayer {
    tensor q_weights, k_weights, v_weights;
    tensor q_biases, k_biases, v_biases;
    tensor o_weights;

    tensor gate_weights, up_weights, down_weights;

    tensor inputNorm_weights, postAttnNorm_weights;
};

struct QwenWeights {
    Config config;
    tensor embedding_weights;
    tensor final_norm_weights;
    std::vector<QwenLayer> layers;
};

QwenWeights loadWeights();

// a is [n, m], b is [m, p]
tensor matMul(const tensor &a, const tensor &b, int n, int m, int p){
    tensor result(n * p, 0.0f);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < m; k++) {
            float av = a[i * m + k];
            if (av == 0.0f) continue;
            for (int j = 0; j < p; j++) {
                result[i * p + j] += av * b[k * p + j];
            }
        }
    }
    return result;
}
tensor addVectors(const tensor &a, const tensor &b){
    tensor sum(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        sum[i] = a[i] + b[i];
    }
    return sum;
}
tensor softmax(const tensor &input);
float silu(float x) {
    return x / (1.0f + expf(-x));
}

// no mean subtraction and no bias, unlike gpt2's layer norm
tensor rmsNorm(const tensor &input, const tensor &weights, float eps);

// rotates q and k in place by position, replaces gpt2's learned position table
void rope(tensor &q, tensor &k, int position, int headDim, float theta);

// 14 query heads share 2 kv heads
tensor groupedQueryAttention(const tensor &embeddings, int numTokens, const QwenLayer &layer,
                             const Config &cfg);

// silu(gate(x)) * up(x), then down
tensor swiglu(const tensor &input, int numTokens, const QwenLayer &layer, const Config &cfg);

tensor transformer(const QwenLayer &layer, int numTokens, const tensor &embeddings,
                   const Config &cfg);

// tokenizer: byte level bpe like gpt2, but a different vocab, a different
// pretokenizer regex and the chatml special tokens
void loadVocab();
std::vector<int> encode(const std::string &text);
std::string decode(const std::vector<int> &ids);
std::string applyChatTemplate(const std::string &userMessage);

tensor qwen(const QwenWeights &w, const std::vector<int> &tokens);
void generate(const QwenWeights &w, const std::string &prompt, int maxNewTokens);

int main() {
    return 0;
}
