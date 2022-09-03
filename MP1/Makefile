NAME = mp1
EXP = data/2/output.raw
IN = data/2/input0.raw,data/2/input1.raw
OUT = actual
TYPE = vector # vector, matrix, or image

include ../CS483.mk

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
