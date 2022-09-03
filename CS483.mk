CC = nvcc
CU = template
WB_PATH = ../libwb
LIBWB = $(WB_PATH)/lib/libwb.so

# Compiling and linking flags.
CCFLAGS = -std=c++11 -rdc=true
LDFLAGS = -std=c++11

####################### Rules #######################

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
