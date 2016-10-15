# For building OpenBLAS.
CC := gcc-4.9
CXX := g++-4.9
FC := gfortran-4.9

LIBRARY_PATH := /opt/cudnn_v5.1/lib64:/usr/local/cuda/lib64

.PHONY: all clean

all:
	CC=$(CC) CXX=$(CXX) FC=$(FC) LIBRARY_PATH=$(LIBRARY_PATH) cargo build --release

clean:
	cargo clean
