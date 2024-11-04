#!/bin/bash

case "$1" in
	"spice")
            for i in {100000000..1000000000..100000000}
            do
                ./spice/build/samples --model=brunel --nsyn=$i
            done
            for i in {100000000..1000000000..100000000}
            do
                ./spice/build/samples --model=brunel+ --nsyn=$i
            done
	    ;;
	"naer")
            cd /home/mattoschmider/naer
	    for i in {100000000..1000000000..100000000}
            do
                ./naer --synapses $i
            done
            for i in {100000000..1000000000..100000000}
            do
                ./naer --plastic --synapses $i
            done
	    ;;
	"spike")
            for i in {100000000..500000000..100000000}
            do
                ./Spike/Build/Examples/Brunel10K --simtime 10.0 --fast --synapses $i
            done
            for i in {100000000..500000000..100000000}
            do
                ./Spike/Build/Examples/Brunel10K --simtime 10.0 --fast --plastic --synapses $i
            done
	    ;;
	*)
            echo "select a simulator to bench with benchmark.sh <spice, naer, spike>"
	    exit 1
	    ;;
esac
