NAME = template
DATASET = 0
TYPE = image # vector, matrix, or image

include ../CS483.mk

EXP = data/$(DATASET)/output.ppm
IN = data/$(DATASET)/input.ppm
OUT = actual.ppm

run_with_sanitizer: $(NAME)
	compute-sanitizer ./$(NAME) -e $(EXP) -i $(IN) -t $(TYPE)
