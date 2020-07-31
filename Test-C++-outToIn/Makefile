
CC=nvcc
CFLAGS=-O0 -g -std=c++11


%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

run.exe: 3d-test.o main.o
		$(CC) -o $@ $^ $(CFLAGS)

