NAME = solution

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
	./$(NAME)

clean:
	rm -f $(NAME) *.o

.PHONY: clean all
