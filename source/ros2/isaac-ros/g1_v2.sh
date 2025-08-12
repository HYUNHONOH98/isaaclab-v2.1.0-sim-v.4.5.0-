#!/bin/bash
shopt -s expand_aliases
source ~/.bashrc   # alias 선언이 여기에 있을 때

# Trap Ctrl+C (SIGINT)
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Kill all background jobs started by this script
    pkill -P $$
    exit 1
}


/workspace/isaaclab/_isaac_sim/python.sh g1_standalone_v2.py


wait