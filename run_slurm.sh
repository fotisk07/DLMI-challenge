#!/usr/bin/env bash
set -e

JOB_NAME=""

# Parse optional --name argument
if [[ "$1" == "--name" ]]; then
  JOB_NAME="$2"
  shift 2
fi

if [ "$#" -lt 2 ]; then
  echo "Usage:"
  echo "  $0 [--name job_name] <prod10|prod20|prod40|prod80> <command> [args...]"
  exit 1
fi

PARTITION="$1"
shift
CMD=("$@")

# Default job name if not provided
if [ -z "$JOB_NAME" ]; then
  JOB_NAME=$(basename "${CMD[0]}")
fi

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err

set -e
echo "Running on partition: ${PARTITION}"
echo "Job name: ${JOB_NAME}"
echo "Command: ${CMD[@]}"

srun ${CMD[@]}
EOF