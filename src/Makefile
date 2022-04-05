CC = gcc
CFLAGS = -O3 -fopenmp
#OPT= -D__MPI__ 
LD = $(CC)
LDFLAGS = $(CFLAGS)

SRC = stream.c
OBJ = $(SRC:.c=.o)
LIBS = -lm

all: stream.exe

stream.exe: $(OBJ)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(OPT) -c $< -o $@

clean:
	rm -rf *~ *.o stream.exe
