NAME = template
DATASET = 0
TYPE = vector # vector, matrix, or image

include ../CS483.mk

EXP = data/$(DATASET)/output.raw
IN = data/$(DATASET)/input.raw

run_with_sanitizer: $(NAME)
	compute-sanitizer ./$(NAME) -e $(EXP) -i $(IN) -t $(TYPE)
