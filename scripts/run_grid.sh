#!/usr/bin/env bash
# Full experiment grid. Run from team2/. Each run fits a modest GPU (teacher is cached).
# Usage: bash scripts/run_grid.sh [main|ablation|teacher|all]
set -euo pipefail
WHAT=${1:-all}
CFG=configs/default.yaml
FRACS="1 5 10"
SEEDS="0 1 2"
ARCHS="resnet18 mobilenetv3_large_100"

if [[ $WHAT == teacher || $WHAT == all ]]; then
  for f in $FRACS; do for s in $SEEDS; do
    python scripts/eval_teacher.py --config $CFG --set data.label_fraction=$f data.split_seed=$s train.seed=$s
  done; done
fi

if [[ $WHAT == main || $WHAT == all ]]; then
  for a in $ARCHS; do for f in $FRACS; do for s in $SEEDS; do
    for m in supervised kd_logit kd_embed pseudo_label rwkd; do
      python -m rwkd.train --config $CFG --set method=$m student.arch=$a \
        data.label_fraction=$f data.split_seed=$s train.seed=$s
    done
  done; done; done
fi

if [[ $WHAT == ablation || $WHAT == all ]]; then
  # ResNet-18, 5% labels, 3 seeds. Each ablation gets its own tag so tables keep them apart.
  for s in $SEEDS; do
    common="method=rwkd student.arch=resnet18 data.label_fraction=5 data.split_seed=$s train.seed=$s"
    python -m rwkd.train --config $CFG --set $common tag=abl_augonly "reliability.components=[aug]"
    python -m rwkd.train --config $CFG --set $common tag=abl_confonly "reliability.components=[conf]"
    python -m rwkd.train --config $CFG --set $common tag=abl_rawweights reliability.transform=raw
    python -m rwkd.train --config $CFG --set $common tag=abl_vote reliability.agreement=vote
    python -m rwkd.train --config $CFG --set $common tag=abl_norelhead student.rel_head=false
    python -m rwkd.train --config $CFG --set $common tag=abl_probeunbalanced teacher.probe_balanced=false
    python -m rwkd.train --config $CFG --set $common tag=abl_zeroshot teacher.target=zeroshot
    python -m rwkd.train --config $CFG --set $common tag=abl_zeroshot_promptaug teacher.target=zeroshot "reliability.components=[prompt,aug]"
    # Optional second teacher (only after its weights + licence are confirmed):
    # python -m rwkd.train --config $CFG --set $common tag=abl_2teachers "teacher.extra_caches=[cache/derm_teacher.pt]" "reliability.components=[aug,conf,teacher]"
  done
fi
