#!/bin/bash

DEVICE=cuda:0
DEFAULT=default
for VERSION in v19;
do
  python src/train.py -d ${DEVICE} -v ${VERSION} -c config/${VERSION}.yaml --default config/${DEFAULT}.yaml
done

