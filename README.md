# Track

A collection of 20 tested projects built from scratch, one module per capstone lesson.

Each module has its own `README.md`, `verify.sh`, tests, and a deterministic offline demo.

## Modules

| # | Module | What it demonstrates | Verify |
|---:|---|---|---|
| 30 | [BPE Tokenizer From Scratch](30-bpe-tokenizer-from-scratch/) | Bytes in, ids out, ids back to the same bytes. Build the tokenizer that every modern text model still starts from. | `cd 30-bpe-tokenizer-from-scratch && ./verify.sh` |
| 31 | [Tokenized Dataset with Sliding Window](31-tokenized-dataset-sliding-window/) | A pretraining run is a function from token ids to gradients. This lesson builds the conveyor that feeds the ids in. | `cd 31-tokenized-dataset-sliding-window && ./verify.sh` |
| 32 | [Token and Positional Embeddings](32-token-positional-embeddings/) | Ids are integers. The model wants vectors. Two lookup tables sit between them, and the choice of the positional one shapes what the model ca | `cd 32-token-positional-embeddings && ./verify.sh` |
| 33 | [Multi-Head Self-Attention](33-multihead-self-attention/) | One linear projection, three views, H parallel heads, one mask. The attention block as the model actually uses it. | `cd 33-multihead-self-attention && ./verify.sh` |
| 34 | [Transformer Block from Scratch](34-transformer-block/) | One block is the unit of every modern decoder LLM. Layer norm, multi head attention, residual, MLP, residual. The pre-LN variant trains stab | `cd 34-transformer-block && ./verify.sh` |
| 35 | [GPT Model Assembly](35-gpt-model-assembly/) | Twelve blocks stacked, a token embedding, a learned position embedding, a final LayerNorm, and a tied language model head. That is the entir | `cd 35-gpt-model-assembly && ./verify.sh` |
| 36 | [Training Loop and Evaluation](36-training-loop-eval/) | A loop that does not measure is a loop that lies. This lesson builds the training loop that drives the GPT model: AdamW with weight decay sp | `cd 36-training-loop-eval && ./verify.sh` |
| 37 | [Loading Pretrained Weights](37-loading-pretrained-weights/) | Training a 124 million parameter model from scratch is a budget decision; loading a published checkpoint is a Tuesday. This lesson loads pre | `cd 37-loading-pretrained-weights && ./verify.sh` |
| 38 | [Capstone Lesson 38: Classifier Fine-Tuning by Head Swap](38-classifier-finetuning/) | Track B's first capstone. A pretrained language model is a stack of self-attention blocks ending in a token-prediction head. When you want s | `cd 38-classifier-finetuning && ./verify.sh` |
| 39 | [Capstone Lesson 39: Instruction Tuning by Supervised Fine-Tuning](39-instruction-tuning-sft/) | A pretrained base model can extend a sequence but cannot follow an instruction. Supervised fine-tuning is the smallest change that fixes thi | `cd 39-instruction-tuning-sft && ./verify.sh` |
| 40 | [Capstone Lesson 40: Direct Preference Optimization from Scratch](40-dpo-from-scratch/) | Reward models and PPO are the classical RLHF stack. DPO collapses that stack into a single supervised loss that fits a policy directly again | `cd 40-dpo-from-scratch && ./verify.sh` |
| 41 | [Capstone Lesson 41: Full Evaluation Pipeline](41-eval-pipeline/) | Training is the part you can monitor with loss curves. Evaluation is the part you have to design. This lesson builds a unified eval pipeline | `cd 41-eval-pipeline && ./verify.sh` |
| 42 | [Large Corpus Downloader](42-large-corpus-downloader/) | Training a language model begins long before the first forward pass. The corpus has to land on disk, decompressed, deduplicated, and address | `cd 42-large-corpus-downloader && ./verify.sh` |
| 43 | [HDF5 Tokenized Corpus](43-hdf5-tokenized-corpus/) | The downloaded corpus has to land in a layout the trainer can stream from at line speed. JSONL on disk does not survive 16 dataloader worker | `cd 43-hdf5-tokenized-corpus && ./verify.sh` |
| 44 | [Cosine LR with Linear Warmup](44-cosine-lr-warmup/) | The learning-rate schedule is the second most important decision after the loss function. AdamW with a cosine decay and a linear warmup is t | `cd 44-cosine-lr-warmup && ./verify.sh` |
| 45 | [Gradient Clipping and Mixed Precision](45-gradient-clipping-amp/) | The optimizer and schedule from the previous lesson assume gradients are sane. They usually are not. A single bad batch can spike the gradie | `cd 45-gradient-clipping-amp && ./verify.sh` |
| 46 | [Gradient Accumulation](46-gradient-accumulation/) | Train at an effective batch you cannot afford, one micro-batch at a time. Scale the loss, hold the optimizer step, and let the gradients pil | `cd 46-gradient-accumulation && ./verify.sh` |
| 47 | [Checkpoint Save and Resume](47-checkpoint-save-resume/) | Train interrupts kill runs; checkpoints let them continue. Save model, optimizer, scheduler, loss history, step counter, and RNG state, atom | `cd 47-checkpoint-save-resume && ./verify.sh` |
| 48 | [Distributed Data Parallel and FSDP from Scratch](48-distributed-fsdp-ddp/) | Multi-rank training is two collectives and one rule. Broadcast the parameters at startup, average the gradients after backward, never let th | `cd 48-distributed-fsdp-ddp && ./verify.sh` |
| 49 | [Language Model Evaluation Harness](49-lm-eval-harness/) | A model that does well on a task you cannot define is a model that does well by accident. The harness is the task definition, the metric, th | `cd 49-lm-eval-harness && ./verify.sh` |

## Requirements

- Python 3.12+ and `uv` for Python modules
- Node 22+ / Bun 1.4 for TypeScript modules
- No API keys, GPU, or network needed for the test suites

## Verify all

```bash
make verify
```

## References

- Capstone source: https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/19-capstone-projects
