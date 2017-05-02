all:
	nvcc -lcuda -lcublas *.cu -o CNN  -arch=compute_20 -Wno-deprecated-gpu-targets

run:
	./CNN
clean:
	rm CNN
