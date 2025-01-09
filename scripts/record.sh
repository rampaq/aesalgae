#!/bin/bash
# records terminal screen when running batched experiments and save to file
#usage: record.sh batch [run_number=0]
runnum="${2:-0}"
cmd="python -u aes-batch.py batches/$1.json -r $runnum"
exec script -efq -c "$cmd" | tee "logs/${1}${runnum}_$(date +%s).log"
