#!/bin/bash

DEVICE=cuda:0
DEFAULT=default

for VERSION in v18;
do
  python src/eval.py -d ${DEVICE} -v ${VERSION}_best -c config/${VERSION}.yaml --default config/${DEFAULT}.yaml --chkpt /home/nas/user/kbh/KungFuMaster/chkpt/${VERSION}/bestmodel.pt
  python src/eval.py -d ${DEVICE} -v ${VERSION}_last -c config/${VERSION}.yaml --default config/${DEFAULT}.yaml --chkpt /home/nas/user/kbh/KungFuMaster/chkpt/${VERSION}/lastmodel.pt
done

