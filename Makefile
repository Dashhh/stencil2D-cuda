CC	= nvcc
CFLAGS	= -arch=sm_30

testerka_stencil:
	$(CC) $(CCFLAGS) testerka_stencil.cu stencil2D.cu -o testerka_stencil

.PHONY: clean
clean:
	rm -f testerka_stencil *.o *~ *.bak
