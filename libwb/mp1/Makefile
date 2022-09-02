CC = nvcc
CU = mp1_from_template
NAME = mp1
WB_LIB = ../lib/libwb.so

EXP = "../../MP1/data/1/output.raw"
IN = "../../MP1/data/1/input0.raw,../../MP1/data/1/input1.raw"
OUT = actual
TYPE = vector # vector, matrix, or image

CCFLAGS = -std=c++11 -rdc=true 
LDFLAGS = -std=c++11 

all: $(NAME)

# Object files.
$(CU).o: $(CU).cu
	$(CC) $(CCFLAGS) -c $(CU).cu -o $@

# The final executable.
$(NAME): $(CU).o $(WB_LIB)
	$(CC) $(LDFLAGS) -o $@ $^

# Convenient run.
run:
	cp $(EXP) expected
	./$(NAME) -e $(EXP) -i $(IN) -o $(OUT) -t $(TYPE)

clean:
	rm -f $(NAME) $(OUT) *.o

.PHONY: clean all