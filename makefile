#Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
#Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
#All rights reserved.
#This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#If there is no config.mk copy over the default version
JUNK := $(shell if [ ! -f "config/config.mk" ];then cp config/defaults.mk config/config.mk; fi)
include config/config.mk

DIRS = exec tensor util ops kernel
SOURCE_FILES = $(foreach dir,$(DIRS),$(wildcard $(dir)/*.cpp))
INCLUDE_FILES = $(foreach dir,$(DIRS),$(wildcard $(dir)/*.hpp)) AcroTensor.hpp
OBJECT_FILES = $(SOURCE_FILES:.cpp=.o)
INCLUDES = $(foreach dir,$(DIRS),-I../$(dir))

ifeq ($(DEBUG),YES)
	CXX_FLAGS = $(INCLUDES) $(CXX_DEBUG)
else
	CXX_FLAGS = $(INCLUDES) $(CXX_OPT)
endif


.SUFFIXES: .cpp .o
.cpp.o:
	cd $(<D); $(CXX) $(CXX_FLAGS) -c $(<F)

all: lib

lib: banner install_dirs libacrotensor.so libacrotensor.a
	cp $(INCLUDE_FILES) inc/

config: FORCE
	@echo "config/config.mk file generated.  Place your build preferences here."

banner:
	@echo ------------------------------------------------------------
	@echo
	@echo "                  Building Acrotensor"
	@echo
	@echo ------------------------------------------------------------

install_dirs:
	if [ ! -d "inc" ];then mkdir inc; fi
	if [ ! -d "lib" ];then mkdir lib; fi
	if [ ! -d "lib/shared" ];then mkdir lib/shared; fi

libacrotensor.so: $(OBJECT_FILES)
	$(CXX) -shared $(OBJECT_FILES) -o lib/shared/libacrotensor.so

libacrotensor.a: $(OBJECT_FILES)
	ar rcs lib/libacrotensor.a $(OBJECT_FILES) 

buildunittest: lib
	cd unittest; make CXX=$(UTILCXX) CXX_FLAGS="$(CXX_FLAGS)" LD_FLAGS="$(UNITTEST_LDFLAGS)"

unittest: buildunittest
	cd unittest; ./test

clean:
	rm -f */*.o */*.o.tgt-nvptx64sm_35-nvidia-linux */*~ *~ 
	rm -rf lib
	rm -rf inc
	cd unittest; make clean

FORCE:
