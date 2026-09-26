#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include "include/json.hpp"
#include<cmath>

using tensor = std::vector<float>;

std::string weightsDir = "../weights/qwen/";

std::vector<std::string> qwenTokens;
std::unordered_map<std::string, int> qwenTokenToId;
std::map<std::pair<std::string, std::string>, int> merges;
std::vector<std::string> specialTokens;

// every byte maps to a printable codepoint so the vocab stays text
std::vector<std::string> byteToUnicode(256);
std::unordered_map<std::string, unsigned char> unicodeToByte;

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

// 8 byte header length, a json index of dtype/shape/byte range, then raw bf16
struct Safetensors {
    struct Entry { std::string dtype; std::vector<int> shape; size_t begin, end; };
    std::string blob;
    size_t dataStart;
    std::unordered_map<std::string, Entry> index;

    void load(const std::string &path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            std::cerr << "missing " << path << " (run modelLoader/qwen.py first)\n";
            std::exit(1);
        }
        blob.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

        uint64_t headerLen;
        std::memcpy(&headerLen, blob.data(), 8);
        dataStart = 8 + headerLen;

        const auto header = nlohmann::json::parse(blob.begin() + 8, blob.begin() + dataStart);
        for (auto it = header.begin(); it != header.end(); ++it) {
            if (it.key() == "__metadata__") continue;
            Entry e;
            e.dtype = it.value()["dtype"];
            e.shape = it.value()["shape"].get<std::vector<int>>();
            e.begin = it.value()["data_offsets"][0];
            e.end   = it.value()["data_offsets"][1];
            index[it.key()] = e;
        }
    }

    const Entry& at(const std::string &name) const {
        auto it = index.find(name);
        if (it == index.end()) {
            std::cerr << "no tensor named " << name << "\n";
            std::exit(1);
        }
        return it->second;
    }

    // bf16 is the top half of a float32, so widening is a shift
    tensor get(const std::string &name) const {
        const Entry &e = at(name);
        const char* src = blob.data() + dataStart + e.begin;
        size_t bytes = e.end - e.begin;

        if (e.dtype == "F32") {
            tensor out(bytes / 4);
            std::memcpy(out.data(), src, bytes);
            return out;
        }
        if (e.dtype == "BF16") {
            tensor out(bytes / 2);
            for (size_t i = 0; i < out.size(); i++) {
                uint16_t half;
                std::memcpy(&half, src + i * 2, 2);
                uint32_t bits = uint32_t(half) << 16;
                std::memcpy(&out[i], &bits, 4);
            }
            return out;
        }
        std::cerr << "unsupported dtype " << e.dtype << " for " << name << "\n";
        std::exit(1);
    }

    // huggingface Linear stores [out, in], matMul wants [in, out]
    tensor getTransposed(const std::string &name) const {
        const Entry &e = at(name);
        tensor src = get(name);
        int rows = e.shape[0], cols = e.shape[1];

        tensor out(src.size());
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                out[c * rows + r] = src[r * cols + c];
            }
        }
        return out;
    }
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

QwenWeights loadWeights() {
    QwenWeights w;
    w.config = loadConfig();

    Safetensors st;
    st.load(weightsDir + "model.safetensors");

    w.embedding_weights  = st.get("model.embed_tokens.weight");
    w.final_norm_weights = st.get("model.norm.weight");

    for (int i = 0; i < w.config.numLayers; i++) {
        std::string p = "model.layers." + std::to_string(i) + ".";
        QwenLayer l;

        l.q_weights = st.getTransposed(p + "self_attn.q_proj.weight");
        l.k_weights = st.getTransposed(p + "self_attn.k_proj.weight");
        l.v_weights = st.getTransposed(p + "self_attn.v_proj.weight");
        l.o_weights = st.getTransposed(p + "self_attn.o_proj.weight");

        // only q, k and v carry biases, o_proj and the mlp have none
        l.q_biases = st.get(p + "self_attn.q_proj.bias");
        l.k_biases = st.get(p + "self_attn.k_proj.bias");
        l.v_biases = st.get(p + "self_attn.v_proj.bias");

        l.gate_weights = st.getTransposed(p + "mlp.gate_proj.weight");
        l.up_weights   = st.getTransposed(p + "mlp.up_proj.weight");
        l.down_weights = st.getTransposed(p + "mlp.down_proj.weight");

        l.inputNorm_weights    = st.get(p + "input_layernorm.weight");
        l.postAttnNorm_weights = st.get(p + "post_attention_layernorm.weight");

        w.layers.push_back(l);
    }
    return w;
}

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
tensor softmax(const tensor &input){
    int n = input.size();
    float mx = *std::max_element(input.begin(), input.end());

    tensor result(n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        result[i] = expf(input[i] - mx);
        sum += result[i];
    }
    for (int i = 0; i < n; i++) {
        result[i] /= sum;
    }
    return result;
}

float silu(float x) {
    return x / (1.0f + expf(-x));
}

// no mean subtraction and no bias, unlike gpt2's layer norm
tensor rmsNorm(const tensor &input, const tensor &weights, float eps) {
    int n = input.size();

    float meanSquare = 0.0f;
    for (auto &i : input) {
        meanSquare += i * i;
    }
    meanSquare /= n;

    float scale = 1.0f / sqrtf(meanSquare + eps);

    tensor output(n);
    for (int i = 0; i < n; i++) {
        output[i] = input[i] * scale * weights[i];
    }
    return output;
}

// position goes in through a rotation instead of gpt2's learned table. pairs
// element i with i + half, not with its neighbour, and the weights depend on it.
void ropeHead(float* x, int position, int headDim, float theta) {
    int half = headDim / 2;

    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(theta, float(2 * i) / headDim);
        float angle = position * freq;
        float c = cosf(angle), s = sinf(angle);

        float x1 = x[i], x2 = x[i + half];
        x[i]        = x1 * c - x2 * s;
        x[i + half] = x2 * c + x1 * s;
    }
}

void rope(tensor &q, tensor &k, int position, int headDim, float theta) {
    ropeHead(q.data(), position, headDim, theta);
    ropeHead(k.data(), position, headDim, theta);
}

// 14 query heads share 2 kv heads
tensor groupedQueryAttention(const tensor &embeddings, int numTokens, const QwenLayer &layer,
                             const Config &cfg) {
    const int H = cfg.hiddenSize, D = cfg.headDim();
    const int kvDim = cfg.numKVHeads * D;   // 128, far narrower than q's 896

    tensor q = matMul(embeddings, layer.q_weights, numTokens, H, H);
    tensor k = matMul(embeddings, layer.k_weights, numTokens, H, kvDim);
    tensor v = matMul(embeddings, layer.v_weights, numTokens, H, kvDim);

    for (int i = 0; i < numTokens; i++) {
        for (int j = 0; j < H; j++)     q[i * H + j]         += layer.q_biases[j];
        for (int j = 0; j < kvDim; j++) k[i * kvDim + j]     += layer.k_biases[j];
        for (int j = 0; j < kvDim; j++) v[i * kvDim + j]     += layer.v_biases[j];
    }

    // q and k get rotated, v does not
    for (int i = 0; i < numTokens; i++) {
        for (int h = 0; h < cfg.numHeads; h++)   ropeHead(&q[i * H + h * D], i, D, cfg.ropeTheta);
        for (int h = 0; h < cfg.numKVHeads; h++) ropeHead(&k[i * kvDim + h * D], i, D, cfg.ropeTheta);
    }

    float scale = 1.0f / sqrtf(float(D));
    tensor output(numTokens * H, 0.0f);

    for (int h = 0; h < cfg.numHeads; h++) {
        int kv = h / cfg.headsPerKV();   // 7 query heads per kv head

        for (int i = 0; i < numTokens; i++) {
            // only earlier tokens, so the causal mask is just the loop bound
            tensor scores(i + 1);
            for (int j = 0; j <= i; j++) {
                float dot = 0.0f;
                for (int d = 0; d < D; d++) {
                    dot += q[i * H + h * D + d] * k[j * kvDim + kv * D + d];
                }
                scores[j] = dot * scale;
            }
            tensor attn = softmax(scores);

            for (int j = 0; j <= i; j++) {
                float a = attn[j];
                for (int d = 0; d < D; d++) {
                    output[i * H + h * D + d] += a * v[j * kvDim + kv * D + d];
                }
            }
        }
    }

    // o_proj has no bias
    return matMul(output, layer.o_weights, numTokens, H, H);
}

// silu(gate(x)) * up(x), then down. no biases anywhere in qwen's mlp.
tensor swiglu(const tensor &input, int numTokens, const QwenLayer &layer, const Config &cfg) {
    const int H = cfg.hiddenSize, I = cfg.intermediateSize;

    tensor gate = matMul(input, layer.gate_weights, numTokens, H, I);
    tensor up   = matMul(input, layer.up_weights,   numTokens, H, I);

    // silu on the gate, then multiply the two elementwise
    for (int i = 0; i < numTokens * I; i++) {
        gate[i] = silu(gate[i]) * up[i];
    }

    return matMul(gate, layer.down_weights, numTokens, I, H);
}

// rmsNorm takes one token, this runs it over the whole block
tensor rmsNormRows(const tensor &input, int numTokens, const tensor &weights, const Config &cfg) {
    tensor output(input.size());
    for (int i = 0; i < numTokens; i++) {
        tensor row(input.begin() + i * cfg.hiddenSize, input.begin() + (i + 1) * cfg.hiddenSize);
        auto normed = rmsNorm(row, weights, cfg.rmsNormEps);
        std::copy(normed.begin(), normed.end(), output.begin() + i * cfg.hiddenSize);
    }
    return output;
}

// norm, sub block, add back. twice.
tensor transformer(const QwenLayer &layer, int numTokens, const tensor &embeddings,
                   const Config &cfg) {
    auto normed = rmsNormRows(embeddings, numTokens, layer.inputNorm_weights, cfg);
    auto attention = groupedQueryAttention(normed, numTokens, layer, cfg);
    tensor residual = addVectors(embeddings, attention);

    auto normedMlp = rmsNormRows(residual, numTokens, layer.postAttnNorm_weights, cfg);
    auto mlp = swiglu(normedMlp, numTokens, layer, cfg);
    return addVectors(residual, mlp);
}

// byte level bpe like gpt2, different vocab and regex, plus chatml tokens
void loadVocab() {
    using json = nlohmann::json;
    std::ifstream f(weightsDir + "tokenizer.json");
    if (!f) {
        std::cerr << "missing tokenizer.json in " << weightsDir << "\n";
        std::exit(1);
    }
    const json data = json::parse(f);

    const auto &vocab = data["model"]["vocab"];
    const auto &added = data["added_tokens"];

    // config pads vocab_size past the real vocab, so size from the ids we see
    int maxId = 0;
    for (auto it = vocab.begin(); it != vocab.end(); ++it) maxId = std::max(maxId, int(it.value()));
    for (const auto &t : added) maxId = std::max(maxId, int(t["id"]));
    qwenTokens.assign(maxId + 1, "");

    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        int id = it.value();
        qwenTokens[id] = it.key();
        qwenTokenToId[it.key()] = id;
    }

    // <|im_start|> and friends live here, not in the vocab
    for (const auto &t : added) {
        int id = t["id"];
        qwenTokens[id] = t["content"];
        qwenTokenToId[t["content"]] = id;
        specialTokens.push_back(t["content"]);
    }

    // newer tokenizer.json writes merges as ["a", "b"], older as "a b"
    int rank = 0;
    for (const auto &m : data["model"]["merges"]) {
        if (m.is_array()) {
            merges[{m[0], m[1]}] = rank++;
        } else {
            std::string s = m;
            auto sp = s.find(' ');
            merges[{s.substr(0, sp), s.substr(sp + 1)}] = rank++;
        }
    }
}
std::string utf8Encode(int codepoint) {
    std::string out;
    if (codepoint < 0x80) {
        out += char(codepoint);
    } else if (codepoint < 0x800) {
        out += char(0xC0 | (codepoint >> 6));
        out += char(0x80 | (codepoint & 0x3F));
    } else {
        out += char(0xE0 | (codepoint >> 12));
        out += char(0x80 | ((codepoint >> 6) & 0x3F));
        out += char(0x80 | (codepoint & 0x3F));
    }
    return out;
}

std::vector<std::string> utf8Chars(const std::string &s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = s[i];
        size_t len = c < 0x80 ? 1 : (c < 0xE0 ? 2 : (c < 0xF0 ? 3 : 4));
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

void initByteEncoder() {
    std::vector<int> bs, cs;
    for (int i = '!'; i <= '~'; i++) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; i++) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; i++) bs.push_back(i);
    cs = bs;

    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    for (size_t i = 0; i < bs.size(); i++) {
        std::string ch = utf8Encode(cs[i]);
        byteToUnicode[bs[i]] = ch;
        unicodeToByte[ch] = (unsigned char)bs[i];
    }
}

std::vector<std::string> bpe(std::vector<std::string> word) {
    while (word.size() > 1) {
        int bestRank = INT_MAX;
        std::pair<std::string, std::string> best;
        for (size_t i = 0; i + 1 < word.size(); i++) {
            auto it = merges.find({word[i], word[i + 1]});
            if (it != merges.end() && it->second < bestRank) {
                bestRank = it->second;
                best = it->first;
            }
        }
        if (bestRank == INT_MAX) break;

        std::vector<std::string> merged;
        for (size_t i = 0; i < word.size();) {
            if (i + 1 < word.size() && word[i] == best.first && word[i + 1] == best.second) {
                merged.push_back(best.first + best.second);
                i += 2;
            } else {
                merged.push_back(word[i]);
                i++;
            }
        }
        word = merged;
    }
    return word;
}

// bpe over ordinary text, no special tokens in here
void encodeChunk(const std::string &text, std::vector<int> &ids) {
    // digits split one at a time. ascii stand in for \p{L} and \p{N}, std::regex
    // has no unicode classes.
    static const std::regex pat(
        R"('s|'t|'re|'ve|'m|'ll|'d|'S|'T|'RE|'VE|'M|'LL|'D)"
        R"(|[^\r\n[:alpha:][:digit:]]?[[:alpha:]]+|[[:digit:]])"
        R"(| ?[^\s[:alpha:][:digit:]]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)");

    auto begin = std::sregex_iterator(text.begin(), text.end(), pat);
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
        std::vector<std::string> word;
        for (unsigned char c : it->str()) word.push_back(byteToUnicode[c]);

        for (const auto &piece : bpe(word)) {
            auto found = qwenTokenToId.find(piece);
            if (found != qwenTokenToId.end()) ids.push_back(found->second);
        }
    }
}

std::vector<int> encode(const std::string &text) {
    std::vector<int> ids;

    // special tokens map straight to their id, bpe would shred them
    size_t pos = 0;
    while (pos < text.size()) {
        size_t best = std::string::npos;
        const std::string* bestTok = nullptr;

        for (const auto &s : specialTokens) {
            size_t at = text.find(s, pos);
            if (at < best) { best = at; bestTok = &s; }
        }
        if (bestTok == nullptr) break;

        if (best > pos) encodeChunk(text.substr(pos, best - pos), ids);
        ids.push_back(qwenTokenToId[*bestTok]);
        pos = best + bestTok->size();
    }
    if (pos < text.size()) encodeChunk(text.substr(pos), ids);

    return ids;
}

std::string decode(const std::vector<int> &ids) {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= (int)qwenTokens.size()) continue;
        for (const auto &ch : utf8Chars(qwenTokens[id])) {
            auto it = unicodeToByte.find(ch);
            if (it != unicodeToByte.end()) out += char(it->second);
            else out += ch;   // special tokens are literal text, not byte encoded
        }
    }
    return out;
}

// chatml. the trailing assistant header is what makes it answer instead of
// carrying on the conversation.
std::string applyChatTemplate(const std::string &userMessage) {
    return "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
           "assistant.<|im_end|>\n<|im_start|>user\n" + userMessage +
           "<|im_end|>\n<|im_start|>assistant\n";
}

// logits for the last position only, that is all sampling needs
tensor qwen(const QwenWeights &w, const std::vector<int> &tokens) {
    const Config &cfg = w.config;
    int n = tokens.size(), H = cfg.hiddenSize;

    // nothing to add for position, rope handles it inside attention
    tensor x(n * H);
    for (int i = 0; i < n; i++) {
        std::copy(w.embedding_weights.begin() + (size_t)tokens[i] * H,
                  w.embedding_weights.begin() + (size_t)(tokens[i] + 1) * H,
                  x.begin() + i * H);
    }

    for (const auto &layer : w.layers) {
        x = transformer(layer, n, x, cfg);
    }

    tensor last(x.end() - H, x.end());
    tensor normed = rmsNorm(last, w.final_norm_weights, cfg.rmsNormEps);

    // tied embeddings, so the output head is the input table again
    int vocab = std::min((int)qwenTokens.size(), cfg.vocabSize);
    tensor logits(vocab);
    for (int t = 0; t < vocab; t++) {
        const float* row = &w.embedding_weights[(size_t)t * H];
        float sum = 0.0f;
        for (int j = 0; j < H; j++) sum += row[j] * normed[j];
        logits[t] = sum;
    }
    return logits;
}

void generate(const QwenWeights &w, const std::string &prompt, int maxNewTokens) {
    std::vector<int> tokens = encode(applyChatTemplate(prompt));
    if (tokens.empty()) {
        std::cerr << "prompt encoded to nothing\n";
        return;
    }

    int imEnd = qwenTokenToId.count("<|im_end|>") ? qwenTokenToId["<|im_end|>"] : -1;
    int eot   = qwenTokenToId.count("<|endoftext|>") ? qwenTokenToId["<|endoftext|>"] : -1;

    // no kv cache, so every token re-runs the whole sequence
    for (int i = 0; i < maxNewTokens && (int)tokens.size() < w.config.maxPos; i++) {
        tensor logits = qwen(w, tokens);
        int next = std::max_element(logits.begin(), logits.end()) - logits.begin();
        if (next == imEnd || next == eot) break;

        tokens.push_back(next);
        std::cout << decode({next}) << std::flush;
    }
    std::cout << "\n";
}

int selfTest() {
    initByteEncoder();
    for (int b = 0; b < 256; b++) assert(unicodeToByte[byteToUnicode[b]] == b);

    // bf16 keeps the exponent and the top 7 mantissa bits, so these are exact
    uint16_t bits = 0x3F80;
    uint32_t wide = uint32_t(bits) << 16;
    float one;
    std::memcpy(&one, &wide, 4);
    assert(one == 1.0f);

    tensor s = softmax({1.0f, 2.0f, 3.0f});
    assert(std::abs(s[0] + s[1] + s[2] - 1.0f) < 1e-5);

    // rmsNorm scales to unit rms when the weights are all one
    tensor r = rmsNorm({3, 4}, {1, 1}, 0.0f);
    assert(std::abs(r[0] * r[0] + r[1] * r[1] - 2.0f) < 1e-4);

    // rope at position 0 is the identity rotation
    tensor x = {1, 2, 3, 4};
    ropeHead(x.data(), 0, 4, 1000000.0f);
    assert(std::abs(x[0] - 1) < 1e-6 && std::abs(x[3] - 4) < 1e-6);

    assert(silu(0.0f) == 0.0f);

    tensor mm = matMul({1, 2, 3, 4, 5, 6}, {1, 0, 0, 1, 1, 1}, 2, 3, 2);
    assert(mm[0] == 4 && mm[1] == 5 && mm[2] == 10 && mm[3] == 11);

    std::cout << "self test passed\n";
    return 0;
}

int main(int argc, char** argv) {
    if (const char* dir = std::getenv("QWEN_WEIGHTS")) weightsDir = std::string(dir) + "/";

    std::string prompt;
    int maxNewTokens = 64;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--test") return selfTest();
        else if (arg == "-n" && i + 1 < argc) maxNewTokens = std::atoi(argv[++i]);
        else prompt = arg;
    }

    initByteEncoder();
    loadVocab();

    std::cerr << "loading weights...\n";
    QwenWeights weights = loadWeights();

    if (!prompt.empty()) {
        generate(weights, prompt, maxNewTokens);
        return 0;
    }

    std::cerr << "ready, " << maxNewTokens << " tokens per prompt. q to quit.\n";
    for (std::string line;;) {
        std::cout << "\ninput: " << std::flush;
        if (!std::getline(std::cin, line) || line == "q") break;
        if (line.empty()) continue;

        std::cout << "output: ";
        generate(weights, line, maxNewTokens);
    }
    return 0;
}
