CC = nvcc
CU = template
WB_PATH = ../libwb
LIBWB = $(WB_PATH)/lib/libwb.so

# Compiling and linking flags.
CCFLAGS = -std=c++11 -rdc=true 
LDFLAGS = -std=c++11 
