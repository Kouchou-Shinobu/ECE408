NAME = mp4
DATASET = 2
TYPE = vector # vector, matrix, or image

include ../CS483.mk

EXP = data/$(DATASET)/output.dat
IN = data/$(DATASET)/input.dat,data/$(DATASET)/kernel.dat

run_with_sanitizer: $(NAME)
	compute-sanitizer ./$(NAME) -e $(EXP) -i $(IN) -t $(TYPE)