#!/bin/bash

REMOTE_DIR=/home/prabodha/project/siddhi-git-dev

echo "scp $1  uom_gpu_cluster3:${REMOTE_DIR}/$1"
scp $1  uom_gpu_cluster3:${REMOTE_DIR}/$1
