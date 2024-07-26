.SUFFIXES:
.PRECIOUS: %.o
.PHONY: all compile checkstyle test clean

CXX = clang++ -std=c++17 -g -Wall -Wextra -Wdeprecated -fsanitize=address
MAIN_BINARIES = $(basename $(wildcard *Main.cpp))
TEST_BINARIES = $(basename $(wildcard *Test.cpp))
LIBS = 
TESTLIBS = -lgtest -lgtest_main -lpthread
OBJECTS = $(addsuffix .o, $(basename $(filter-out %Main.cpp %Test.cpp, $(wildcard *.cpp))))

all: compile checkstyle test

compile: $(MAIN_BINARIES) $(TEST_BINARIES)

#checkstyle:
#	clang-format-14 --dry-run -Werror *.h *.cpp

test: $(TEST_BINARIES)
	for T in $(TEST_BINARIES); do ./$$T || exit; done

%.o: %.cpp *.h
	$(CXX) -c $<

%Main: %Main.o $(OBJECTS)
	$(CXX) -o $@ $^ $(LIBS)

%Test: %Test.o $(OBJECTS)
	$(CXX) -o $@ $^ $(LIBS) $(TESTLIBS)

clean:
	rm -f *Main
	rm -f *Test
	rm -f *.o
	rm -f *.bin

format:
	clang-format-14 -i *.cpp *.h