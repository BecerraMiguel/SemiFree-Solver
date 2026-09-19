# Build and run the unit/regression tests of the semi-free solver.
#
# Usage (from the repository root):
#   make -C src/semifree_boundary/tests            compile all tests into build/
#   make -C src/semifree_boundary/tests quick      compile + run the 6 fast tests
#   make -C src/semifree_boundary/tests test       compile + run all 8 tests (slow, see below)
#   make -C src/semifree_boundary/tests run-<name> run a single test, e.g. run-cutcell_geometry
#   make -C src/semifree_boundary/tests clean      remove the test binaries
#
# Optional: TIMEOUT=<seconds> caps every test run (a capped test is reported as such), e.g.
#   make -C src/semifree_boundary/tests test TIMEOUT=600
#
# Tests read their data from tests/fixtures/ (a small mini repository root with cases/,
# configs/ and reference data), so they are always run from that directory.
#
# Slow tests: `regularization` and `cutcell_singularity_global` evaluate the plasma
# integral over full meshes (the latter also launches build/semifree_solver with a
# 120 s cap), so `test` can take a long time. `quick` skips them.

CXX      ?= g++
CXXFLAGS := -O3 -fopenmp -std=c++11
INCLUDES := -I../../third_party -I..
LDLIBS   := -lm

BUILD     := ../../../build
FIX_BUILD := ../../../../build

FAST := adaptive_quadrature closedform_greg cutcell_geometry grid_jphi_reconstruction \
        singularity_kernels singularity_subtraction
SLOW := cutcell_singularity_global regularization
ALL  := $(FAST) $(SLOW)

DEPS := $(wildcard ../*.h) ../SemiFree_Solver.cpp

RUN = $(if $(TIMEOUT),timeout $(TIMEOUT) ,)

.PHONY: all quick test clean solver

all: $(addprefix $(BUILD)/test_,$(ALL))

$(BUILD)/test_%: test_%.cpp $(DEPS)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDLIBS)

# cutcell_singularity_global launches the real solver binary as one of its checks.
solver:
	$(MAKE) -C ../../.. build/semifree_solver

run-%: $(BUILD)/test_%
	cd fixtures && $(RUN)$(FIX_BUILD)/test_$*

define RUN_LIST
	@fail=0; \
	for t in $(1); do \
	  echo "=================== test_$$t"; \
	  (cd fixtures && $(RUN)$(FIX_BUILD)/test_$$t); rc=$$?; \
	  if [ $$rc -eq 0 ]; then echo ">>> $$t: OK"; \
	  elif [ $$rc -eq 124 ]; then echo ">>> $$t: TIMEOUT"; fail=$$((fail+1)); \
	  else echo ">>> $$t: FAILED (exit $$rc)"; fail=$$((fail+1)); fi; \
	done; \
	echo; echo "$$fail test(s) did not pass"; exit $$fail
endef

quick: $(addprefix $(BUILD)/test_,$(FAST))
	$(call RUN_LIST,$(FAST))

test: solver $(addprefix $(BUILD)/test_,$(ALL))
	$(call RUN_LIST,$(ALL))

clean:
	rm -f $(addprefix $(BUILD)/test_,$(ALL))
