#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>
#include <cmath>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <unordered_map>
#include "include/json.hpp"

using tensor = std::vector<float>;
constexpr double EPSILON = 0.00001;

const int N_EMBD = 768;
const int N_HEAD = 12;
const int N_LAYERS = 12;
const int N_CTX = 1024;
const int N_VOCAB = 50257;

std::string weightsDir = "../weights/gpt2/";

struct TransformerInput {
    // attention qkv w & b
    tensor q_weights, k_weights, v_weights;
    tensor q_biases, k_biases, v_biases;

    // attention output projection
    tensor attnProj_weights, attnProj_biases;

    // mlp
    tensor l1_weights, l2_weights;
    tensor l1_biases, l2_biases;

    // layer norm
    tensor lnAttention_weights, lnAttenion_biases;
    tensor lnMlp_weights, lnMlp_biases;

    TransformerInput(
        tensor q_w, tensor k_w, tensor v_w,
        tensor q_b, tensor k_b, tensor v_b,
        tensor proj_w, tensor proj_b,
        tensor l1_w, tensor l2_w,
        tensor l1_b, tensor l2_b,
        tensor lnAtt_w, tensor lnAtt_b,
        tensor lnMlp_w, tensor lnMlp_b
    )
        : q_weights(q_w), k_weights(k_w), v_weights(v_w),
          q_biases(q_b), k_biases(k_b), v_biases(v_b),
          attnProj_weights(proj_w), attnProj_biases(proj_b),
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

// gpt2 maps every byte to a printable codepoint so the vocab stays text
std::vector<std::string> byteToUnicode(256);
std::unordered_map<std::string, unsigned char> unicodeToByte;

std::string getTokenFromTokenId(int tokenId) {
    auto token = gpt2tokens[tokenId];
    return token;
}

tensor matMul(const tensor& a, const tensor& b, int n, int m, int p) {
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

// row-wise layer norm over a [numTokens, N_EMBD] block
tensor layerNormRows(const tensor &input, int numTokens, const tensor &weights, const tensor &biases) {
    tensor output(numTokens * N_EMBD);
    for(int i = 0; i < numTokens; i++){
        tensor row(input.begin() + i * N_EMBD, input.begin() + (i + 1) * N_EMBD);
        auto normed = layerNorm(row, weights, biases);
        std::copy(normed.begin(), normed.end(), output.begin() + i * N_EMBD);
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

// weights are stored [inputs, neurons] like huggingface Conv1D
tensor forwardPass(const tensor &weights, const tensor &biases, const tensor &inputs, bool use_gelu = false) {
    int countNeurons = biases.size();
    tensor output(biases.begin(), biases.end());

    for(int j = 0; j < inputs.size(); j++){
        float in = inputs[j];
        for(int i = 0; i < countNeurons; i++){
            output[i] += weights[j * countNeurons + i] * in;
        }
    }

    if(use_gelu){
        for(int i = 0; i < countNeurons; i++){
            output[i] = gelu(output[i]);
        }
    }

    return output;
}

tensor attention(
    const tensor &embeddings, int numTokens, int embedDim, int headDim,
    const tensor &qWeights, const tensor &kWeights, const tensor &vWeights,
    const tensor &qBiases, const tensor &kBiases, const tensor &vBiases
) {
    // project embeddings into Q, K, V
    tensor qProj = matMul(embeddings, qWeights, numTokens, embedDim, headDim);
    tensor kProj = matMul(embeddings, kWeights, numTokens, embedDim, headDim);
    tensor vProj = matMul(embeddings, vWeights, numTokens, embedDim, headDim);

    // add biases
    for(int i = 0; i < numTokens; i++){
        for(int j = 0; j < headDim; j++){
            qProj[i * headDim + j] += qBiases[j];
            kProj[i * headDim + j] += kBiases[j];
            vProj[i * headDim + j] += vBiases[j];
        }
    }

    // attention scores: Q * K^T / sqrt(headDim)
    tensor kTranspose = transpose(kProj, numTokens, headDim);
    tensor scores = matMul(qProj, kTranspose, numTokens, headDim, numTokens);

    float scale = 1.0f / sqrt(headDim);
    for(int i = 0; i < numTokens * numTokens; i++){
        scores[i] *= scale;
    }

    // causal mask: a token may not look ahead
    for(int i = 0; i < numTokens; i++){
        for(int j = i + 1; j < numTokens; j++){
            scores[i * numTokens + j] = -1e9f;
        }
    }

    // softmax over each row
    tensor attnWeights(numTokens * numTokens);
    for(int i = 0; i < numTokens; i++){
        tensor row(scores.begin() + i * numTokens, scores.begin() + (i + 1) * numTokens);
        tensor softmaxRow = softmax(row);
        for(int j = 0; j < numTokens; j++){
            attnWeights[i * numTokens + j] = softmaxRow[j];
        }
    }

    // weighted sum: attn * V
    tensor output = matMul(attnWeights, vProj, numTokens, numTokens, headDim);
    return output;
}

tensor multiHeadAttention(
    const tensor &embeddings, int numTokens,
    const TransformerInput &layer
) {
    const int HEAD_DIM = N_EMBD / N_HEAD;

    // run each head, writing its slice into the [numTokens, N_EMBD] concat
    tensor allHeads(numTokens * N_EMBD);
    for(int h = 0; h < N_HEAD; h++) {
        // slice Q, K, V weights for this head: columns [h*HEAD_DIM, (h+1)*HEAD_DIM)
        tensor qh_w, kh_w, vh_w;
        for(int i = 0; i < N_EMBD; i++){
            for(int j = 0; j < HEAD_DIM; j++){
                qh_w.push_back(layer.q_weights[i * N_EMBD + h * HEAD_DIM + j]);
                kh_w.push_back(layer.k_weights[i * N_EMBD + h * HEAD_DIM + j]);
                vh_w.push_back(layer.v_weights[i * N_EMBD + h * HEAD_DIM + j]);
            }
        }

        // slice Q, K, V biases for this head: [HEAD_DIM]
        tensor qh_b(layer.q_biases.begin() + h * HEAD_DIM, layer.q_biases.begin() + (h + 1) * HEAD_DIM);
        tensor kh_b(layer.k_biases.begin() + h * HEAD_DIM, layer.k_biases.begin() + (h + 1) * HEAD_DIM);
        tensor vh_b(layer.v_biases.begin() + h * HEAD_DIM, layer.v_biases.begin() + (h + 1) * HEAD_DIM);

        // single head attention
        tensor headOut = attention(embeddings, numTokens, N_EMBD, HEAD_DIM,
                                   qh_w, kh_w, vh_w, qh_b, kh_b, vh_b);

        for(int i = 0; i < numTokens; i++){
            std::copy(headOut.begin() + i * HEAD_DIM, headOut.begin() + (i + 1) * HEAD_DIM,
                      allHeads.begin() + i * N_EMBD + h * HEAD_DIM);
        }
    }

    // project concatenated heads through output projection
    tensor output = matMul(allHeads, layer.attnProj_weights, numTokens, N_EMBD, N_EMBD);
    for(int i = 0; i < numTokens; i++){
        for(int j = 0; j < N_EMBD; j++){
            output[i * N_EMBD + j] += layer.attnProj_biases[j];
        }
    }
    return output;
}

tensor mlp(const tensor &input, int numTokens, int dimensions, const TransformerInput &layer) {
    tensor result(numTokens * dimensions);
    for(int i = 0; i < numTokens; i++) {
        // extract this token's embedding
        tensor tokenEmb(input.begin() + i * dimensions, input.begin() + (i + 1) * dimensions);

        // up projection: [dimensions] x [dimensions, HIDDEN] -> [HIDDEN], then gelu
        tensor hiddenOut = forwardPass(layer.l1_weights, layer.l1_biases, tokenEmb, true);

        // down projection: [HIDDEN] x [HIDDEN, dimensions] -> [dimensions]
        tensor out = forwardPass(layer.l2_weights, layer.l2_biases, hiddenOut, false);

        // store result
        for(int j = 0; j < dimensions; j++){
            result[i * dimensions + j] = out[j];
        }
    }
    return result;
}

tensor transformer(const TransformerInput &input, int numTokens, const tensor &embeddings) {
    // layer norm before attention
    auto layerNormEmbeddings = layerNormRows(embeddings, numTokens, input.lnAttention_weights, input.lnAttenion_biases);

    // multi-head attention
    auto attentionResult = multiHeadAttention(layerNormEmbeddings, numTokens, input);

    // residual connection
    tensor residual = addVectors(embeddings, attentionResult);

    // layer norm before MLP
    auto normedMLP = layerNormRows(residual, numTokens, input.lnMlp_weights, input.lnMlp_biases);

    // MLP
    auto mlpResult = mlp(normedMLP, numTokens, N_EMBD, input);

    // residual connection after MLP
    return addVectors(residual, mlpResult);
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

// split a utf8 string into its characters
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

void loadVocab() {
    using json = nlohmann::json;
    std::ifstream f(weightsDir + "tokenizer/tokenizer.json");
    if (!f) {
        std::cerr << "cannot open tokenizer.json under " << weightsDir << "\n";
        std::exit(1);
    }
    const json data = json::parse(f);

    gpt2tokens.assign(N_VOCAB, "");
    for (auto it = data["model"]["vocab"].begin(); it != data["model"]["vocab"].end(); ++it) {
        int id = it.value();
        gpt2tokens[id] = it.key();
        gpt2TokenToId[it.key()] = id;
    }
    for (const auto &tok : data["added_tokens"]) {
        int id = tok["id"];
        gpt2tokens[id] = tok["content"];
        gpt2TokenToId[tok["content"]] = id;
    }

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

// merge the lowest-ranked pair repeatedly until none of the pairs are known
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

std::vector<int> encode(const std::string &text) {
    // gpt2 splits on \p{L} and \p{N}, which std::regex has no idea about, so this is the
    // ascii version. non-ascii still encodes fine, it just splits in different places.
    static const std::regex pat(
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)");

    std::vector<int> ids;
    auto begin = std::sregex_iterator(text.begin(), text.end(), pat);
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
        std::vector<std::string> word;
        for (unsigned char c : it->str()) word.push_back(byteToUnicode[c]);

        for (const auto &piece : bpe(word)) {
            auto found = gpt2TokenToId.find(piece);
            if (found != gpt2TokenToId.end()) ids.push_back(found->second);
        }
    }
    return ids;
}

std::string decode(const std::vector<int> &ids) {
    std::string out;
    for (int id : ids) {
        for (const auto &ch : utf8Chars(getTokenFromTokenId(id))) {
            auto it = unicodeToByte.find(ch);
            if (it != unicodeToByte.end()) out += char(it->second);
        }
    }
    return out;
}

Gptweights loadWeights() {
    // reading 1.4GB with >> float takes minutes, so slurp the file and walk it with strtof
    auto readFile = [](const std::string& path, tensor& vec) {
        std::ifstream stream(path, std::ios::binary);
        if (!stream) {
            std::cerr << "missing weight file: " << path << "\n";
            std::exit(1);
        }
        std::string buf((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
        vec.reserve(buf.size() / 11 + 1);

        const char* p = buf.c_str();
        char* end;
        for (;;) {
            float v = std::strtof(p, &end);
            if (end == p) break;
            vec.push_back(v);
            p = end;
        }
    };

    std::vector<TransformerInput> layers;

    for(int i = 0; i < N_LAYERS; i++) {
        std::string prefix = weightsDir + "transformer.h." + std::to_string(i) + ".";

        // layer norm 1
        tensor ln1_w, ln1_b;
        readFile(prefix + "ln_1.weight.txt", ln1_w);
        readFile(prefix + "ln_1.bias.txt", ln1_b);

        // fused qkv, stored [N_EMBD, 3*N_EMBD] so q/k/v are column blocks not row blocks
        tensor qkv_w, qkv_b;
        readFile(prefix + "attn.c_attn.weight.txt", qkv_w);
        readFile(prefix + "attn.c_attn.bias.txt", qkv_b);

        tensor q_w(N_EMBD * N_EMBD), k_w(N_EMBD * N_EMBD), v_w(N_EMBD * N_EMBD);
        for(int r = 0; r < N_EMBD; r++){
            const float* row = &qkv_w[r * 3 * N_EMBD];
            std::copy(row, row + N_EMBD, q_w.begin() + r * N_EMBD);
            std::copy(row + N_EMBD, row + 2 * N_EMBD, k_w.begin() + r * N_EMBD);
            std::copy(row + 2 * N_EMBD, row + 3 * N_EMBD, v_w.begin() + r * N_EMBD);
        }

        tensor q_b(qkv_b.begin(), qkv_b.begin() + N_EMBD);
        tensor k_b(qkv_b.begin() + N_EMBD, qkv_b.begin() + 2 * N_EMBD);
        tensor v_b(qkv_b.begin() + 2 * N_EMBD, qkv_b.end());

        // attention projection
        tensor attn_proj_w, attn_proj_b;
        readFile(prefix + "attn.c_proj.weight.txt", attn_proj_w);
        readFile(prefix + "attn.c_proj.bias.txt", attn_proj_b);

        // layer norm 2
        tensor ln2_w, ln2_b;
        readFile(prefix + "ln_2.weight.txt", ln2_w);
        readFile(prefix + "ln_2.bias.txt", ln2_b);

        // mlp c_fc (up projection)
        tensor fc_w, fc_b;
        readFile(prefix + "mlp.c_fc.weight.txt", fc_w);
        readFile(prefix + "mlp.c_fc.bias.txt", fc_b);

        // mlp c_proj (down projection)
        tensor proj_w, proj_b;
        readFile(prefix + "mlp.c_proj.weight.txt", proj_w);
        readFile(prefix + "mlp.c_proj.bias.txt", proj_b);

        layers.push_back(TransformerInput(
            q_w, k_w, v_w, q_b, k_b, v_b,
            attn_proj_w, attn_proj_b,
            fc_w, proj_w, fc_b, proj_b,
            ln1_w, ln1_b, ln2_w, ln2_b
        ));
    }

    // token and positional embeddings
    tensor tok_w, pos_w;
    readFile(weightsDir + "transformer.wte.weight.txt", tok_w);
    readFile(weightsDir + "transformer.wpe.weight.txt", pos_w);

    // final layer norm
    tensor lnf_w, lnf_b;
    readFile(weightsDir + "transformer.ln_f.weight.txt", lnf_w);
    readFile(weightsDir + "transformer.ln_f.bias.txt", lnf_b);

    return Gptweights(tok_w, pos_w, lnf_b, lnf_w, layers);
}

float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

tensor getTokenEmbedding(const Gptweights &w, const std::vector<int> &tokens) {
    int n = tokens.size();
    tensor embeddings(n * N_EMBD);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < N_EMBD; j++){
            embeddings[i * N_EMBD + j] =
                w.embedding_weights[tokens[i] * N_EMBD + j] + w.positionalEmbeddings[i * N_EMBD + j];
        }
    }
    return embeddings;
}

// logits for the last position only, that is all sampling needs
tensor gpt(const Gptweights &w, const std::vector<int> &tokens) {
    int n = tokens.size();
    tensor x = getTokenEmbedding(w, tokens);

    for(const auto &layer : w.transformer_weights){
        x = transformer(layer, n, x);
    }

    tensor last(x.end() - N_EMBD, x.end());
    tensor normed = layerNorm(last, w.final_weights, w.final_bias);

    // output head is the tied token embedding: logits[v] = wte[v] . h
    tensor logits(N_VOCAB);
    for(int v = 0; v < N_VOCAB; v++){
        float sum = 0.0f;
        const float* row = &w.embedding_weights[(size_t)v * N_EMBD];
        for(int j = 0; j < N_EMBD; j++) sum += row[j] * normed[j];
        logits[v] = sum;
    }
    return logits;
}

int selfTest() {
    initByteEncoder();

    // byte encoder round trips every byte
    for(int b = 0; b < 256; b++) assert(unicodeToByte[byteToUnicode[b]] == b);

    // softmax normalises and preserves ordering
    tensor s = softmax({1.0f, 2.0f, 3.0f});
    assert(std::abs(s[0] + s[1] + s[2] - 1.0f) < 1e-5);
    assert(s[2] > s[1] && s[1] > s[0]);

    // layer norm gives zero mean, unit variance when scale=1 bias=0
    tensor ln = layerNorm({1, 2, 3, 4}, {1, 1, 1, 1}, {0, 0, 0, 0});
    float mean = 0;
    for(float v : ln) mean += v;
    assert(std::abs(mean) < 1e-4);

    // matMul: [2,3] x [3,2]
    tensor mm = matMul({1, 2, 3, 4, 5, 6}, {1, 0, 0, 1, 1, 1}, 2, 3, 2);
    assert(mm[0] == 4 && mm[1] == 5 && mm[2] == 10 && mm[3] == 11);

    // forwardPass reads weights as [inputs, neurons]
    tensor fp = forwardPass({1, 2, 3, 4}, {0, 0}, {1, 1});
    assert(fp[0] == 4 && fp[1] == 6);

    // causal mask: token 0 cannot see token 1, so its output is v[0] exactly
    tensor emb = {1, 0, 0, 1};
    tensor eye = {1, 0, 0, 1}, zero = {0, 0};
    tensor at = attention(emb, 2, 2, 2, eye, eye, eye, zero, zero, zero);
    assert(std::abs(at[0] - 1.0f) < 1e-5 && std::abs(at[1] - 0.0f) < 1e-5);

    // bpe round trip against the real vocab
    loadVocab();
    std::string text = "Hello world, the quick brown fox!";
    auto ids = encode(text);
    assert(!ids.empty());
    assert(decode(ids) == text);

    std::cout << "self test passed (" << ids.size() << " tokens for the bpe check)\n";
    return 0;
}

void generate(const Gptweights &weights, const std::string &prompt, int maxNewTokens) {
    std::vector<int> tokens = encode(prompt);
    if (tokens.empty()) {
        std::cerr << "prompt encoded to nothing\n";
        return;
    }

    std::cout << prompt << std::flush;

    // there is no kv cache, so every token re-runs the whole sequence from scratch
    for (int i = 0; i < maxNewTokens && (int)tokens.size() < N_CTX; i++) {
        tensor logits = gpt(weights, tokens);
        int next = std::max_element(logits.begin(), logits.end()) - logits.begin();
        if (next == 50256) break;  // <|endoftext|>

        tokens.push_back(next);
        std::cout << decode({next}) << std::flush;
    }
    std::cout << "\n";
}

int main(int argc, char** argv) {
    if (const char* dir = std::getenv("GPT2_WEIGHTS")) weightsDir = std::string(dir) + "/";

    std::string prompt;
    int maxNewTokens = 20;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--test") return selfTest();
        else if (arg == "-n" && i + 1 < argc) maxNewTokens = std::atoi(argv[++i]);
        else prompt = arg;
    }

    initByteEncoder();
    loadVocab();

    std::cerr << "loading weights...\n";
    Gptweights weights = loadWeights();

    // one prompt on the command line, otherwise loop so the weights stay loaded
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
 