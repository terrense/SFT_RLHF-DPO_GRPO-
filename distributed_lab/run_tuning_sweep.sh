#!/bin/bash
# DeepSpeed tuning single-variable sweep. Qwen2.5-0.5B full-param, 4 GPU, 60 steps.
# Each run varies ONE knob from ZeRO-2 base. Measures peak mem/card + wall time.
set -e
export PATH=/root/miniconda3/bin:$PATH
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/distributed_lab
mkdir -p logs tune_out
SUM=logs/tuning_summary.txt
echo "name|dsconfig|micro|accum|gradckpt|globalBatch|peakMB|sec|samples_per_s" > $SUM

run_one () {
  NAME=$1; DS=$2; MICRO=$3; ACCUM=$4; GC=$5
  GLOBAL=$((MICRO * ACCUM * 4))
  echo "===== RUN $NAME  ds=$DS micro=$MICRO accum=$ACCUM gc=$GC global=$GLOBAL ====="
  # build yaml
  sed -e "s#__DS__#ds_configs/$DS#" bench_train.yaml > tune_out/$NAME.yaml
  sed -i "s#per_device_train_batch_size: 8#per_device_train_batch_size: $MICRO#" tune_out/$NAME.yaml
  sed -i "s#gradient_accumulation_steps: 1#gradient_accumulation_steps: $ACCUM#" tune_out/$NAME.yaml
  sed -i "s#gradient_checkpointing: false#gradient_checkpointing: $GC#" tune_out/$NAME.yaml
  sed -i "s#output_dir: .*#output_dir: /root/autodl-tmp/distributed_lab/tune_out/$NAME#" tune_out/$NAME.yaml
  # peak mem poller
  PEAKF=logs/peak_$NAME.txt; echo 0 > $PEAKF
  ( while true; do
      M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
      P=$(cat $PEAKF); if [ "$M" -gt "$P" ]; then echo $M > $PEAKF; fi
      sleep 1
    done ) &
  POLL=$!
  T0=$(date +%s)
  llamafactory-cli train tune_out/$NAME.yaml > logs/train_$NAME.log 2>&1 || echo "TRAIN_FAILED $NAME"
  T1=$(date +%s)
  kill $POLL 2>/dev/null || true
  SEC=$((T1 - T0))
  PEAK=$(cat $PEAKF)
  SAMPLES=$((GLOBAL * 60))
  SPS=$(python3 -c "print(round($SAMPLES/$SEC,1))" 2>/dev/null || echo NA)
  echo "$NAME|$DS|$MICRO|$ACCUM|$GC|$GLOBAL|$PEAK|$SEC|$SPS" >> $SUM
  echo "  -> peak=${PEAK}MB  time=${SEC}s  sps=$SPS"
  sleep 3
}

# single-variable sweep from ZeRO-2 base (micro8/accum1/gc off/bucket200M/overlap on)
run_one base        z2_base.json         8 1 false
run_one micro2      z2_base.json         2 4 false
run_one gradckpt    z2_base.json         8 1 true
run_one smallbucket z2_smallbucket.json  8 1 false
run_one bigbucket   z2_bigbucket.json    8 1 false
run_one nooverlap   z2_nooverlap.json    8 1 false

echo '===== SWEEP DONE ====='
cat $SUM
