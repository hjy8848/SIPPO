# JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG

This repository contains the source code for the paper **"JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG"**.

JADE (Joint Agentic Dynamic Execution) is a unified framework that formulates dynamic Agentic RAG as a cooperative multi-agent game. By optimizing the Planner and Executors jointly using a single shared backbone, JADE addresses the "strategic-operational mismatch" found in existing decoupled or static architectures.

This implementation is based on the [verl](https://github.com/volcengine/verl) library.

## Installation

Please use the following commands to set up the environment:

```
# Install dependencies (vLLM, SGLang, Megatron-Core)
bash ./scripts/install_vllm_sglang_mcore_0.7.sh

# Install JADE in editable mode
pip install --no-deps -e .

# Ensure numpy compatibility
pip install "numpy<2.0"
```

Training
To start the training process (Joint Agentic Dynamic Optimization), run the following script:

```
bash run_without_debugger.sh
```
The training process utilizes PPO to optimize the shared parameters across the Planner and Executors, aggregating heterogeneous transitions into a Unified Experience Buffer.
