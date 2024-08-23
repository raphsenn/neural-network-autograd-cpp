.SUFFIXES:
.PRECIOUS: %.o
.PHONY: all compile checkstyle test clean format

CXX = clang++ -std=c++17 -g -Wall -Wextra -Wdeprecated -fsanitize=address
SRC_DIR = src
BIN_DIR = bin
MAIN_SOURCES = $(wildcard $(SRC_DIR)/*Main.cpp)
TEST_SOURCES = $(wildcard $(SRC_DIR)/*Test.cpp)
LIBS =
TESTLIBS = -lgtest -lgtest_main -lpthread
OBJECTS = $(addprefix $(BIN_DIR)/, $(notdir $(addsuffix .o, $(basename $(filter-out %Main.cpp %Test.cpp, $(wildcard $(SRC_DIR)/*.cpp))))))

all: compile checkstyle test

compile: $(BIN_DIR) $(MAIN_SOURCES:$(SRC_DIR)/%.cpp=$(BIN_DIR)/%) $(TEST_SOURCES:$(SRC_DIR)/%.cpp=$(BIN_DIR)/%)

checkstyle:
	clang-format --dry-run -Werror $(SRC_DIR)/*.h $(SRC_DIR)/*.cpp

test: $(TEST_SOURCES:$(SRC_DIR)/%.cpp=$(BIN_DIR)/%)
	for T in $(TEST_SOURCES:$(SRC_DIR)/%.cpp=$(BIN_DIR)/%); do ./$$T || exit; done

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp $(SRC_DIR)/*.h | $(BIN_DIR)
	$(CXX) -c $< -o $@

$(BIN_DIR)/%Main: $(BIN_DIR)/%Main.o $(OBJECTS) | $(BIN_DIR)
	$(CXX) -o $@ $^ $(LIBS)

$(BIN_DIR)/%Test: $(BIN_DIR)/%Test.o $(OBJECTS) | $(BIN_DIR)
	$(CXX) -o $@ $^ $(LIBS) $(TESTLIBS)

clean:
	rm -rf $(BIN_DIR)

format:
	clang-format -i $(SRC_DIR)/*.cpp $(SRC_DIR)/*.h