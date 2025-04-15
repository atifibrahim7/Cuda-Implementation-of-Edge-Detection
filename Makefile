CC=nvcc
LIBS=
SOURCE_DIR= .
BIN_DIR= .
CFLAGS= -O1 -g
LDFLAGS= -lm
OBJS=$(SOURCE_DIR)/canny_edge.o $(SOURCE_DIR)/hysteresis.o $(SOURCE_DIR)/pgm_io.o
EXEC= canny
INCS= -I.
CSRCS= $(SOURCE_DIR)/canny_edge.cu \
       $(SOURCE_DIR)/hysteresis.cu \
       $(SOURCE_DIR)/pgm_io.cu

# PIC=pics/pic_small.pgm
# PIC=pics/pic_medium.pgm
PIC=pics/pic_large.pgm
# PIC=pics/pic_tiny.pgm

all: canny

canny: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(EXEC) $(PIC)
	./$(EXEC) $(PIC) 2.5 0.25 0.5
# 			sigma tlow thigh

gprof: CFLAGS += -pg
gprof: LDFLAGS += -pg
gprof: clean all
	./$(EXEC) $(PIC) 2.0 0.5 0.5
	gprof -b ./$(EXEC) gmon.out > gprof_$(EXEC).txt
	gprof2dot gprof_$(EXEC).txt > profile.dot
	dot -Tpng profile.dot -o profile.png

clean:
	@-rm -rf canny $(OBJS) gmon.out gprof_$(EXEC).txt profile.dot profile.png

.PHONY: clean comp exe run all gprof