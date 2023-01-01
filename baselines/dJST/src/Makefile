OBJS=	strtokenizer.o dataset.o utils.o polya_fit_simple.o math_func.o model.o inference.o
MAIN=	djst
CC = g++
DEBUG = -g
#CFLAGS = -Wall -c $(DEBUG)
CFLAGS = -Wall -c
LFLAGS = -Wall $(DEBUG)
LIBJST = libjst.a

ifeq ($(BUILD),debug)   
# "Debug" build - no optimization, and debugging symbols
CFLAGS += -O0 -g -fPIC
else
# # "Release" build - optimization, and no debug symbols
CFLAGS += -O2 -s -DNDEBUG -fPIC
endif

all: djst

debug:
	make "BUILD=debug"

djst: main.o ${LIBJST}
	$(CC) -o djst main.o -L. -ljst

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp -I`pwd`
	#g++ -c main.cpp

${LIBJST}: ${OBJS}
	ar cr ${LIBJST} ${OBJS}
        
strtokenizer.o:	strtokenizer.h strtokenizer.cpp
	$(CC) $(CFLAGS) -o strtokenizer.o strtokenizer.cpp

dataset.o:	dataset.h dataset.cpp
	$(CC) $(CFLAGS) -o dataset.o dataset.cpp

utils.o:	utils.h utils.cpp
	$(CC) $(CFLAGS) -o utils.o utils.cpp

polya_fit_simple.o:	polya_fit_simple.h polya_fit_simple.cpp
	$(CC) $(CFLAGS) -o polya_fit_simple.o polya_fit_simple.cpp
	
math_func.o:	math_func.h math_func.cpp
	$(CC) $(CFLAGS) -o math_func.o math_func.cpp
	
model.o:	model.h model.cpp
	$(CC) $(CFLAGS) -o model.o model.cpp

inference.o:	inference.h inference.cpp
	$(CC) $(CFLAGS) -o inference.o inference.cpp
	
clean:
	rm $(OBJS) 
	rm $(MAIN)
	rm ${LIBJST}
