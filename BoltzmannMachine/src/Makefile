CC				= g++
OBJS			= bm.o bmModule.o
PROGRAM			= a.out

all:			$(PROGRAM)

$(PROGRAM):		$(OBJS)
				$(CC) $(OBJS) $(LDFLAGS) $(LIBS) -o $(PROGRAM)

clean:;			rm -f *.o *~ $(PROGRAM)
