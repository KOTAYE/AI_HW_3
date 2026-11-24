#!/bin/bash
# XLM-R on all non-val folds, its stack, then xAI on the 150k mDeBERTa (must finish before the 14:39 power cut).
cd "$(dirname "$0")/../src"; export PYTHONIOENCODING=utf-8
run() { name=$1; shift; echo "$(date +%H:%M) start $name" >> ../logs/queue.log; python "$@" > ../logs/$name.log 2>&1 && echo "$(date +%H:%M) done $name" >> ../logs/queue.log || echo "$(date +%H:%M) FAILED $name" >> ../logs/queue.log; }
run xlmr_full train_encoder.py --model FacebookAI/xlm-roberta-base --name xlmr_full --train-set full --epochs 2 --bs 8 --accum 2 --lr 3e-5
run stack_mdeb150k_xlmrfull stack.py stack --enc mdeberta_150k xlmr_full --tfidf baseline_full --feats stylo --name stack_mdeb150k_xlmrfull --submit
run xai_fusion xai.py fusion --model models/mdeberta_150k
run xai_ig xai.py ig --model models/mdeberta_150k
