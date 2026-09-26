# Inference Engine for Qwen

GPT-2 and Qwen2.5-0.5B on the CPU, no ML libraries.

## Running

Download the weights:

```
cd modelLoader
uv run gpt2.py     # ~1.4GB of text files into weights/gpt2/
uv run qwen.py     # safetensors into weights/qwen/
```

Build:

```
cd inference
make
```

GPT-2 completes text, Qwen answers as a chat model:

```
./gpt2                       # interactive, q to quit
./gpt2 "Once upon a time"    # single prompt, then exit
./gpt2 -n 40                 # 40 tokens instead of the default 20

./qwen                       # interactive, q to quit
./qwen "What is 2+2?"
./qwen -n 128                # default is 64

./gpt2 --test                # self tests, no weights needed
./qwen --test
```

Weights are looked up at `../weights/gpt2/` and `../weights/qwen/`, so run from
`inference/` or set `GPT2_WEIGHTS` / `QWEN_WEIGHTS`.

Neither one caches keys and values yet, so every token replays the whole sequence.
Qwen takes a few seconds per token.
