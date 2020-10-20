#!/bin/bash
for input_s in 44
do
for d4 in 2
do
for d1 in 84 100 168 250
do
for drop1 in 0
do
for d2 in 250 500 750
do
for drop2 in 0
do
for d3 in 500 750 1000
do
for l1_h in 0
do
for l2_h in 0
do
for l1_l in 0
do
for l2_l in 0
do
for layers in 1 2 3
do
for epochs in 750 1000 1500
do
#echo $k1 $k2 $k3 $k4 $k5 $k6 $k7 $k8 $k9 $k10 $k11 $k12 $k13
python 4_DNN_training.py $input_s $d4 $d1 $drop1 $d2 $drop2 $d3 $l1_h $l2_h $l1_l $l2_l $layers $epochs
done
done
done
done
done
done
done
done
done
done
done
done
done
