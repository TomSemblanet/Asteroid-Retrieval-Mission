#!/bin/bash

for file in $1/*
do 
	python -m scripts.utils.post_process $1/$file
done