#!/bin/bash
# Waits for mdeberta_150k (PID given as $1), stacks it, then trains XLM-R on 150k and stacks both.
cd "$(dirname "$0")/../src"; export PYTHONIOENCODING=utf-8
while tasklist | grep -q " $1 "; do sleep 60; done
echo "$(date +%H:%M) done mdeberta_150k" >> ../logs/queue.log
python stack.py stack --enc mdeberta_150k xlmr --tfidf baseline_full --feats stylo --name stack_mdeb150k_xlmr --submit > ../logs/stack_mdeb150k_xlmr.log 2>&1
echo "$(date +%H:%M) start xlmr_150k" >> ../logs/queue.log
python train_encoder.py --model FacebookAI/xlm-roberta-base --name xlmr_150k --train-set full --max-train 150000 --epochs 2 --bs 8 --accum 2 --lr 3e-5 > ../logs/xlmr_150k.log 2>&1 \
  && echo "$(date +%H:%M) done xlmr_150k" >> ../logs/queue.log || echo "$(date +%H:%M) FAILED xlmr_150k" >> ../logs/queue.log
python stack.py stack --enc mdeberta_150k xlmr_150k --tfidf baseline_full --feats stylo --name stack_mdeb150k_xlmr150k --submit > ../logs/stack_mdeb150k_xlmr150k.log 2>&1
python stack.py stack --enc mdeberta_150k xlmr_150k mdeberta xlmr --tfidf baseline_full --feats stylo --name stack_4enc --submit > ../logs/stack_4enc.log 2>&1
echo "$(date +%H:%M) night queue finished" >> ../logs/queue.log
