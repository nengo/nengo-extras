#!/usr/bin/env bash

NAME=$0
COMMAND=$1

function usage {
    echo "usage: $NAME <command>"
    echo
    echo "  install  Install dependencies for testing and collecting coverage"
    echo "  run      Run pytest and collect coverage"
    echo "  upload   Upload coverage to codecov.io"
    exit 1
}

if [[ "$COMMAND" == "install" ]]; then
    conda install --quiet jupyter keras matplotlib numba "numpy<1.16" pillow scipy tensorflow theano
    pip install coverage "pytest<4.0.0"
    pip install -e .
elif [[ "$COMMAND" == "run" ]]; then
    coverage run -m pytest nengo_extras -v --duration 20 --plots && coverage report
elif [[ "$COMMAND" == "upload" ]]; then
    eval "bash <(curl -s https://codecov.io/bash)"
else
    if [[ -z "$COMMAND" ]]; then
        echo "Command required"
    else
        echo "Command $COMMAND not recognized"
    fi
    echo
    usage
fi
