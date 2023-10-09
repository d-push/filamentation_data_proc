#!/bin/bash

sudo python3.11 setup.py install --record files.txt
cat files.txt | sudo xargs sudo rm -rf
rm files.txt
