#Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
#Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
#All rights reserved.
#This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

# Serial compiler
ACRO_BASEDIR = ..

CXX         = clang++
CXX_FLAGS   = -O3 -fopenmp=libomp -std=c++11 -stdlib=libc++
CCC = $(CXX) $(CXX_FLAGS)
LD_FLAGS = -Wl,-rpath,$(ACRO_BASEDIR)

DIRS = exec tensor util kernel
SRCDIRS = $(foreach dir,$(DIRS),$(ACRO_BASEDIR)/$(dir))
INCLUDES = -I. $(foreach dir,$(SRCDIRS),-I$(dir)) -I../inc

SOURCE_FILES = unit_test_main.cpp $(sort $(wildcard ./*/*.cpp))
HEADER_FILES = catch.hpp
OBJECT_FILES = $(SOURCE_FILES:.cpp=.o)
ACRO_LIB = $(ACRO_BASEDIR)/lib/shared/libacrotensor.so

.SUFFIXES: .cpp .o
.cpp.o:
	$(CCC) -c $(<D)/$(<F) $(INCLUDES) -o $(<D)/$(<F:.cpp=.o)

test: banner $(OBJECT_FILES)
	$(CXX) $(OBJECT_FILES) $(INCLUDES) -lrt $(ACRO_LIB) $(LD_FLAGS) -o test

banner:
	@echo ------------------------------------------------------------
	@echo
	@echo "              Building Acrotensor Unittests"
	@echo
	@echo ------------------------------------------------------------

$(OBJECT_FILES): $(HEADER_FILES)

clean:
	rm -f *.o */*.o */*~ *~ *.core *.v3breakpoints test
