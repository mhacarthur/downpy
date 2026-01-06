#!/bin/bash

source ~/anaconda3/bin/activate AXE

cd python

product="$1"

if [[ "$product" == "PERSIANN" ]]; then
    python 2_GAMMA.py -pr PERSIANN -tr 1dy -ys 2002 -ye 2012
elif [[ "$product" == "SM2RAIN" ]]; then
    python 2_GAMMA.py -pr SM2RAIN -tr 1dy -ys 2002 -ye 2012
else
    echo "Error: Producto no válido. Usa 'PERSIANN' o 'SM2RAIN'."
    exit 1
fi




