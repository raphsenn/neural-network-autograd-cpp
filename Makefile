
compile:
	clang++ -c -std=c++17 Matrix.cpp
	clang++ -c -std=c++17 MatrixTest.cpp
	clang++ -o MatrixTest Matrix.o MatrixTest.o -lgtest -lgtest_main

test:
	./MatrixTest

clean:
	rm -f *.o
	rm -f MatrixTest