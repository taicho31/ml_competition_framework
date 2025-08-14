#!/bin/bash
COMPETITION="atma_202507"

sudo chmod 666 /var/run/docker.sock
cp /home/taichi/my_work/python/data_competition_prevwork/template/model.py /home/taichi/my_work/python/$COMPETITION/main/common_script

#docker compose up -d --build
