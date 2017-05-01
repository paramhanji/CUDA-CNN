all:
	nvcc -lcuda -lcublas *.cu -o CNN  -arch=compute_20

clean:
	rm CNN
