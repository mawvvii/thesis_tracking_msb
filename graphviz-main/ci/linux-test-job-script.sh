#!/usr/bin/env bash

set -e
set -o pipefail
set -u
set -x

ci/install-packages.sh
export GV_VERSION=$( cat GRAPHVIZ_VERSION )
python3 -m pytest --strict-markers --verbose --verbose --junit-xml=report.xml \
  ci/tests.py tests
