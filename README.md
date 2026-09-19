# Inference Engine for Qwen

## Running

Download the weights (writes ~1.4GB of text files into `weights/`):

```
cd modelLoader
uv run main.py
```

Build and run:

```
cd inference
g++ -O2 -march=native -std=c++17 -o gpt2 main.cc

./gpt2                       # interactive, q to quit
./gpt2 -n 40                 # 40 tokens per prompt instead of the default 20
./gpt2 "Once upon a time"    # single prompt, then exit
./gpt2 --test                # self test, no weights needed
```

The weights are looked up at `../weights/`, so run from `inference/` or set
`GPT2_WEIGHTS=/path/to/weights`.

Currently it works with gpt2 weights - will be adapting it to support qwen0.5b.
