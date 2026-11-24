#!/usr/bin/env bash
# Sequential GPU queue on the local RTX 5060 (8 GB). Each run logs to logs/<name>.log; a failed run does not stop the queue.
cd "$(dirname "$0")/../src"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 TRANSFORMERS_VERBOSITY=error
run() { name=$1; shift; echo "$(date +%H:%M) start $name"; python "$@" > ../logs/$name.log 2>&1 && echo "$(date +%H:%M) done $name" || echo "$(date +%H:%M) FAILED $name"; }
run mdeberta      train_encoder.py --model microsoft/mdeberta-v3-base --name mdeberta --epochs 2 --bs 4 --accum 4
run xlmr          train_encoder.py --model FacebookAI/xlm-roberta-base --name xlmr --epochs 2 --bs 8 --accum 2 --lr 3e-5
run xlmr_head     train_encoder.py --model FacebookAI/xlm-roberta-base --name xlmr_head --epochs 2 --bs 8 --accum 2 --lr 3e-5 --head-only
run ppl_qwen15    ppl.py --model Qwen/Qwen2.5-1.5B --name qwen15 --max-tokens 512 --bs 8
