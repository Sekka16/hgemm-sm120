all:
	nvcc -O3 -lineinfo -Xptxas -v -arch=sm_120 test_hgemm.cu -lcublas -o test_hgemm 

clean:
	rm -f test_hgemm