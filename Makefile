# Build system for the semi-free boundary solver.
#
# Usage (from the repository root):
#   make          build build/semifree_solver
#   make clean    remove build/semifree_solver
#
# OpenMP (-fopenmp) is required for the multi-threaded psi/B evaluation loops.
# Without it g++ silently ignores the OpenMP pragmas and the solver runs
# single-threaded, so it is always enabled here.
# To verify a binary is linked against OpenMP:  ldd build/semifree_solver | grep gomp

CXX      ?= g++
CXXFLAGS := -O3 -fopenmp -std=c++11
INCLUDES := -Isrc/third_party
LDLIBS   := -lm

SEMIFREE_SRC  := src/semifree_boundary/SemiFree_Solver.cpp
SEMIFREE_DEPS := $(wildcard src/semifree_boundary/*.h)

.PHONY: all clean

all: build/semifree_solver

build/semifree_solver: $(SEMIFREE_SRC) $(SEMIFREE_DEPS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(SEMIFREE_SRC) $(LDLIBS)

clean:
	rm -f build/semifree_solver
