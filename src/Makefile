
BOOST_INCLUDES = -I/opt/local/include/boost -I/home-local/walch/boost_1_57_0/include
BOOST_LIBDIR = -L/opt/local/lib -L/home-local/walch/boost_1_57_0/lib

FREEIMAGE_INCLUDES = -I/opt/local/include -I/home-local/walch/FreeImage/Dist
FREEIMAGE_LIBDIR = -L/opt/local/lib -L/home-local/walch/FreeImage/Dist

# 32 bit
MARCH=-m32 -msse2 -mfpmath=sse

# 64 bit
MARCH=-m64

# change this to -g3 for debug builds
OPT=-O2

#########################################################################
# nothing below this line should need to be changed (usually)
#########################################################################

INCLUDE_DIRECTORIES = $(sort $(dir $(wildcard ./*/)))
INCLUDE_DIRS = $(addprefix -I,$(INCLUDE_DIRECTORIES))

ARCH = $(shell uname)

CC=gcc
CXX=g++
LD=$(CXX) $(OPT) $(MARCH)
INCLUDE=-I. $(INCLUDE_DIRS) $(BOOST_INCLUDES) $(FREEIMAGE_INCLUDES)
WARN=-w
CWD=$(shell pwd)
CXXFLAGS=$(OPT) $(MARCH) $(INCLUDE) $(WARN) $(DEFS) -std=gnu++0x
CCFLAGS=$(CXXFLAGS)
LIBS=$(BOOST_LIBDIR) $(FREEIMAGE_LIBDIR) -lm

ifeq ($(ARCH),Linux)
  LIBS += -lboost_system -lboost_thread -lboost_filesystem -lboost_program_options -lfreeimage
  LIBS += -lOpenCL
endif
#ifeq ($(ARCH),OpenBSD)
#  LIBS += -l
#endif
ifeq ($(ARCH),Darwin)
  LIBS +=-lboost_system-mt -lboost_thread-mt -lboost_filesystem-mt -lboost_program_options-mt -lfreeimage
  LIBS += -framework OpenCL
endif

CXXSRCS = $(wildcard core/*.cpp utils/*.cpp geometry/*.cpp accelerators/*.cpp)
CXXSRCS +=  $(wildcard idevices/*.cpp engines/*.cpp film/*.cpp lighttree/*.cpp)
OBJS += $(addprefix objs/, $(subst /,_,$(CXXSRCS:.cpp=.o)))

default: CXXFLAGS += -DDISABLE_OPENGL
default: dirs bin/dprt 

#Enable OpenGL window
ifeq ($(ARCH),Darwin)
dprtgl: LIBS += -framework OpenGL -framework GLUT
endif
dprtgl: dirs bin/dprtgl 

dirs:
	/bin/mkdir -p ./bin ./objs

.PHONY: dirs

objs/engines_%.o: engines/%.cpp
	@echo "Building object $@"
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/core_%.o: core/%.cpp
	@echo "Building object $@"
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/accelerators_%.o: accelerators/%.cpp
	@echo "Building object $@"
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/idevices_%.o: idevices/%.cpp
	@echo "Building object $@"
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/film_%.o: film/%.cpp
	@echo "Building object $@"
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/geometry_%.o: geometry/%.cpp
	@echo "Building object $@"
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/lighttree_%.o: lighttree/%.cpp
	@echo "Building object $@"
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/utils_%.o: utils/%.cpp
	@echo "Building object $@"
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

objs/dprt.o: main/dprt.cpp
	@echo "Building object $@"
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

bin/dprt: objs/dprt.o $(OBJS)
	@echo "Linking $@"
	@$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

bin/dprtgl: objs/dprt.o $(OBJS)
	@echo "Linking $@"
	@$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f ./objs/* ./bin/*


print-%:
	@echo $* = $($*)


