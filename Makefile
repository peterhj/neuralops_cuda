LIBRARY_PATH := /opt/cudnn_v5.1/lib64:/usr/local/cuda/lib64

.PHONY: all clean

all:
	LIBRARY_PATH=$(LIBRARY_PATH) cargo build --release

clean:
	cargo clean
