#!/usr/bin/env bash

set -e
set -o pipefail
set -u
set -x

logfile=$(mktemp)
ci/build.sh |& tee $logfile
echo "$CI_JOB_NAME-warnings `grep -c 'warning:' $logfile`" | tee metrics.txt
