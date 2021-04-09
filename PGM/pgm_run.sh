#!/bin/sh
g++ pgm.cpp -o pgm.out
./pgm.out
gnuplot -persist pgm.plt
