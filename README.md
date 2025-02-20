# aespa
This repository contains the code for the NeurIPS 2024 paper [**Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers**](https://arxiv.org/pdf/2402.08958). 

The current release includes the following features:
  - An efficient implementation of the proposed aespa algorithm: `aespa.py`
  - Compressing all models from the OPT, BLOOM, LLaMA, LLaMA2, LLaMA3 families to 2/3/4 bits: `main.py`
  - Evaluating the perplexity of quantized models on several language generation tasks: `main.py`

## Dependencies
 - `torch`: tested on v2.1.0
 - `transformers`: tested on v4.43.2
 - `datasets`: tested on v2.20.0

## aespa options
 - `block_v`: whether to apply block-wise objective (Eq.(17)) for the value projection 
    - For query and key projections, block-wise objectives (Eqs. (21) and (22)) are always used.
 - `use_zfold`: whether to use Z-Fold in computing quantization parameters (Z-Fold: https://aclanthology.org/2023.emnlp-main.892)
 - `optq_init`: whether to update full-precision weights based on OPTQ before applying AdaRound (OPTQ: https://arxiv.org/abs/2210.17323)
    - `act_order`: whether to apply OPTQ heuristic
 - `learn_rounding`: whether to learn weight-rounding policy based on AdaRound
    - `lr`, `round_weight`, `round_weight_qkv`, `num_iters`: AdaRound hyperparameters

## Examples
 - OPT Model Quantization (results w/ H100 GPU - wikitext2: 69.813, ptb-new: 100.23, c4-new: 56.377)

  ```
  python main.py --model_path facebook/opt-125m --calib_data c4 --nsamples 128 --seqlen 2048 --seed 0 --w_bits 2 --block_v --use_zfold --optq_init --act_order --learn_rounding
  ```
