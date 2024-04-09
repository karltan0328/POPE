#/bin/bash

echo "hello!"

echo "python3 train-linemod.py"
python3 train-linemod.py

echo "python3 train-onepose.py"
python3 train-onepose.py

echo "python3 train-onepose++.py"
python3 train-onepose++.py
