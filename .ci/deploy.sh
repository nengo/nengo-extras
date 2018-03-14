#!/usr/bin/env bash

# Automatically generated by nengo-bones, do not edit this file directly

NAME=$0
COMMAND=$1
STATUS=0  # used to exit with non-zero status if any command fails

exe() {
    echo "\$ $*";
    # remove empty spaces from args
    args=( "$@" )
    for i in "${!args[@]}"; do
      [ -n "${args[$i]}" ] || unset "args[$i]"
    done
    "${args[@]}" || { echo -e "\033[1;31mCOMMAND '${args[0]}' FAILED\033[0m"; STATUS=1; }
}

if [[ ! -e nengo_extras ]]; then
    echo "Run this script from the root directory of this repository"
    exit 1
fi

if [[ "$COMMAND" == "install" ]]; then
    :


    exe pip install "check-manifest>=0.37" "collective.checkdocs>=0.2" "pygments>=2.3.1"
    exe pip install -e .
elif [[ "$COMMAND" == "before_script" ]]; then
    :
elif [[ "$COMMAND" == "script" ]]; then

    exe check-manifest
    exe python setup.py checkdocs
    if [[ "$TRAVIS_TAG" == "" ]]; then
        TAG=v$(cut -d'-' -f3 <<<"$TRAVIS_BRANCH")
    else
        TAG="$TRAVIS_TAG"
    fi
    exe python -c "from nengo_extras import version; \
        assert version.dev is None, 'this is a dev version'"
    exe python -c "from nengo_extras import version; \
        assert 'v' + version.version == '$TAG', 'version does not match tag'"
    exe python -c "from nengo_extras import version; \
        assert any(line.startswith(version.version) \
        and 'unreleased' not in line \
        for line in open('CHANGES.rst').readlines()), \
        'changelog not updated'"

elif [[ "$COMMAND" == "before_cache" ]]; then
    :
elif [[ "$COMMAND" == "after_success" ]]; then
    :
elif [[ "$COMMAND" == "after_failure" ]]; then
    :
elif [[ "$COMMAND" == "before_deploy" ]]; then
    :
elif [[ "$COMMAND" == "after_deploy" ]]; then
    :
elif [[ "$COMMAND" == "after_script" ]]; then
    :
elif [[ -z "$COMMAND" ]]; then
    echo "$NAME requires a command like 'install' or 'script'"
else
    echo "$NAME does not define $COMMAND"
fi

if [[ "$COMMAND" != "script" && -n "$TRAVIS" && "$STATUS" -ne "0" ]]; then
    travis_terminate "$STATUS"
fi
exit "$STATUS"
