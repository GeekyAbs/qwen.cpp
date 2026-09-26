CXX ?= g++
CXXFLAGS ?= -O2 -march=native -std=c++17

all: gpt2 qwen

gpt2: gpt2.cc include/json.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

qwen: qwen.cc include/json.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test: all
	./gpt2 --test
	./qwen --test

clean:
	rm -f gpt2 qwen

.PHONY: all test clean
