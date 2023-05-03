#!/bin/bash
PATH_FRAMS="/home/adam/Framsticks/Framsticks50rc19/"
OPT="vertpos"
PATH_OUT="/home/adam/thesis/VGAE/baseline/"
# PATH_FRAMS=#"/home/inf131778/Framsticks50rc19/"
# OPT="vertpos"
# PATH_OUT= #"/home/inf131778/VGAE/baseline/"
cd framspy
python3 -u -m  evolalg.examples.standard -path $PATH_FRAMS -opt $OPT -path_out $PATH_OUT  -id 1