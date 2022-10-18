CC = nvcc
CU = template
WB_PATH = ../libwb
LIBWB = $(WB_PATH)/lib/libwb.so

# Compiling and linking flags.
CCFLAGS = -std=c++11 -rdc=true -arch=sm_86
LDFLAGS = -std=c++11 -arch=sm_86

# args for the executables.
EXP = data/$(DATASET)/output.raw
IN = data/$(DATASET)/input0.raw,data/$(DATASET)/input1.raw
OUT = actual

# Debug option
debug =
ifeq ($(debug), true)
CCFLAGS += -D __DEBUG__
endif

########################### Rules ###########################

all: $(NAME)

# Object files.
$(CU).o: $(CU).cu
	$(CC) $(CCFLAGS) -c $(CU).cu -I $(WB_PATH) -o $@

# The final executable.
$(NAME): $(CU).o $(LIBWB)
	$(CC) $(LDFLAGS) -o $@ $^

# The final executable, but with the default name.
template: $(CU).o $(LIBWB)
	$(CC) $(LDFLAGS) -o $@ $^

# Convenient run.
run_single: $(NAME)
	cp $(EXP) expected
	./$(NAME) -e $(EXP) -i $(IN) -o $(OUT) -t $(TYPE)

run: template
	bash run_datasets

clean:
	rm -rf $(NAME) $(OUT) \
			*.o expected template bench

.PHONY: clean all run run_single
