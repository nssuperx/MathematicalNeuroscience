#!/bin/sh
g++ am.cpp -o am.out
./am.out
gnuplot -persist am.plt
