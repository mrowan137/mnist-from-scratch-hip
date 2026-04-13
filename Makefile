CXX = hipcc
CXXFLAGS = -O3 -std=c++17 -Wall -Werror

mnist: mnist.cpp
	${CXX} ${CXXFLAGS} mnist.cpp -o mnist

clean:
	rm -rf mnist
