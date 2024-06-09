#!/bin/bash

rm -rf ./results/*
for env in "pendulum" "cartpole" "cheetah"
do
  for i in 1 2 3
  do
    python3 main.py --env-name $env --seed $i --baseline &
    python3 main.py --env-name $env --seed $i --no-baseline &
    python3 main.py --env-name $env --seed $i --ppo &
  done
done