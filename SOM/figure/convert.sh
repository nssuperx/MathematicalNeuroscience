#!/bin/bash

for f in *.gif
do
    convert $f +adjoin ${f%.*}.png
done
