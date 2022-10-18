#!/usr/bin/env bash

target="$1"
if test -z target
then
  echo 'Need to provide a target MP to run testcases on.'
  exit -1
fi

# Go to the target working dir.
cd ${target}
if test $? -ne 0
then
  echo '$target does not seem to exist.'
  exit -1
fi
# Make things up.
make template

bash run_datasets
# Back to basics.
cd ..
