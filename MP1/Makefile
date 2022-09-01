CC = nvcc
CU = template
NAME = mp1
WB_PATH = ../libwb
LIBWB = $(WB_PATH)/lib/libwb.so

EXP = data/1/output.raw
IN = data/1/input0.raw,data/1/input1.raw
OUT = actual
TYPE = vector # vector, matrix, or image

CCFLAGS = -std=c++11 -rdc=true 
LDFLAGS = -std=c++11 

all: $(NAME)

# Object files.
$(CU).o: $(CU).cu
	$(CC) $(CCFLAGS) -c $(CU).cu -I $(WB_PATH) -o $@

# The final executable.
$(NAME): $(CU).o $(LIBWB)
	$(CC) $(LDFLAGS) -o $@ $^

# Convenient run.
run:
	cp $(EXP) expected
	./$(NAME) -e $(EXP) -i $(IN) -o $(OUT) -t $(TYPE)

clean:
	rm -f $(NAME) $(OUT) \
			*.o expected

.PHONY: clean all
