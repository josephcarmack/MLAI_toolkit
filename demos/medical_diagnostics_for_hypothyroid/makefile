# bDemo makefile
# Author: Joseph Carmack

.SUFFIXES:
.SUFFIXES: .cpp .o .dep

SHELL = /bin/sh
TARGET_NAME = hypothyroid_demo.app
DBG_NAME = dbg$(TARGET_NAME)
CXX = g++
CFLAGS = -Wall -std=c++11
LFLAGS = 
CFLAGS_OPT = $(CFLAGS) -O3
CFLAGS_DBG = $(CFLAGS) -g -D_DEBUG
SRC_PATH = ../../src/
OBJ_PATH_OPT = ../../src/object_files/opt/
DEP_PATH_OPT = ../../src/make_rules/opt/
OBJ_PATH_DBG = ../../src/object_files/dbg/
DEP_PATH_DBG = ../../src/make_rules/dbg/
MAIN_OBJ_OPT = optmain
MAIN_OBJ_DBG = dbgmain

# list files from the src directory that the demo depends on here
SRC_FILES = \
			error.cpp \
			json.cpp \
			kdtree.cpp \
			knn.cpp \
			matrix.cpp \
			rand.cpp \
			string.cpp \
			vec.cpp \

# build some lists of files for use in below Make rules
# use the 'printLists' rules to see their values
#
TEMP_OBJ_OPT = $(SRC_FILES:%.cpp=%.o)
OBJ_OPT = $(TEMP_OBJ_OPT:%=$(OBJ_PATH_OPT)%)
TEMP_OPT_DEP = $(SRC_FILES:.cpp=.dep)
OPT_DEP_FILES = $(TEMP_OPT_DEP:%=$(DEP_PATH_OPT)%)

TEMP_OBJ_DBG = $(SRC_FILES:.cpp=.o)
OBJ_DBG = $(TEMP_OBJ_DBG:%=$(OBJ_PATH_DBG)%)
TEMP_DBG_DEP = $(SRC_FILES:.cpp=.dep)
DBG_DEP_FILES = $(TEMP_DBG_DEP:%=$(DEP_PATH_DBG)%)

# usage rule: explains make targets
.PHONY : usage
usage:
	@echo
	@echo "HOW TO USE THIS MAKEFILE"
	@echo "************************"
	@echo
	@echo "    MAKE CMD --------- (CMD DESCRIPTION)"
	@echo "    **********************************************************"
	@echo "    make opt --------- (build an optimized executable)"
	@echo "    make dbg --------- (build a debug executable)"
	@echo "    make clean ------- (clean up demo .o files and executables)"
	@echo "    make usage"
	@echo "    or just 'make' --- (show these instructions again)"
	@echo "    ***********************************************************"
	@echo

# rule for building the optimized executable
opt : $(MAIN_OBJ_OPT).o $(OBJ_OPT) 
	@echo
	@echo "            **************************************"
	@echo "                LINKING OPTIMIZED OBJECT FILES"
	@echo "            **************************************"
	@echo
	$(CXX) -o $(TARGET_NAME) $(OBJ_OPT) $(MAIN_OBJ_OPT).o $(LFLAGS)
	@echo
	@echo

# rule for building the debug executable
dbg : $(MAIN_OBJ_DBG).o $(OBJ_DBG) 
	@echo
	@echo "            **************************************"
	@echo "                LINKING OPTIMIZED OBJECT FILES"
	@echo "            **************************************"
	@echo
	$(CXX) -o $(DBG_NAME) $(OBJ_DBG) $(MAIN_OBJ_DBG).o $(LFLAGS)
	@echo
	@echo

# clean up demo files
clean : 
	rm -f $(MAIN_OBJ_OPT).o $(MAIN_OBJ_OPT).dep $(MAIN_OBJ_DBG).o \
		$(MAIN_OBJ_DBG).dep $(TARGET_NAME) $(DBG_NAME)

# clean up demo files and delete src object files 
fclean: 
	rm -f $(MAIN_OBJ_OPT).o $(MAIN_OBJ_OPT).dep $(MAIN_OBJ_DBG).o \
		$(MAIN_OBJ_DBG).dep $(OBJ_OPT) $(OBJ_DBG) $(TARGET_NAME) \
		$(DBG_NAME)

# rule for printing values stored in list variables
# (for better understanding how the makefile works)
printLists:
	@echo
	@echo "TEMP_OBJ_OPT = $(TEMP_OBJ_OPT)"
	@echo
	@echo "OBJ_OPT = $(OBJ_OPT)"
	@echo
	@echo "TEMP_OPT_DEP = $(TEMP_OPT_DEP)"
	@echo
	@echo "OPT_DEP_FILES = $(OPT_DEP_FILES)"
	@echo
	@echo "TEMP_OBJ_DBG = $(TEMP_OBJ_DBG)"
	@echo
	@echo "OBJ_DBG = $(OBJ_DBG)"
	@echo
	@echo "TEMP_DBG_DEP = $(TEMP_DBG_DEP)"
	@echo
	@echo "DBG_DEP_FILES = $(DBG_DEP_FILES)"
	@echo




###################################################################
#       RULES FOR AUTO-GENERATING DEPENDENCY RULES                #
###################################################################




# generate and include rule for compiling main.cpp's optimized object file
optmain.dep : main.cpp
	@echo -n "opt" > $(MAIN_OBJ_OPT).dep
	@$(CXX) -MM main.cpp >> $(MAIN_OBJ_OPT).dep
	@echo "	$(CXX) $(CFLAGS_OPT) -c main.cpp -o $(MAIN_OBJ_OPT).o" >> $(MAIN_OBJ_OPT).dep
-include optmain.dep

# generate and include rule for compiling main.cpp's debug object file
$(MAIN_OBJ_DBG).dep : main.cpp
	@echo -n "dbg" > $(MAIN_OBJ_DBG).dep
	@$(CXX) -MM main.cpp >> $(MAIN_OBJ_DBG).dep
	@echo "	$(CXX) $(CFLAGS_DBG) -c main.cpp -o $(MAIN_OBJ_DBG).o" >> $(MAIN_OBJ_DBG).dep
-include $(MAIN_OBJ_DBG).dep

# rule for automatically generating OPTIMIZED object file rules (.o files) using the compiler for src files
$(OPT_DEP_FILES) : $(DEP_PATH_OPT)%.dep: $(SRC_PATH)%.cpp
	@echo "generating OPTIMIZED make rule and recipe for $*.o and storing it in $@ using the $(CXX) compiler."
	@echo -n "$(OBJ_PATH_OPT)" > $@
	@$(CXX) -MM $< >> $@
	@echo "	$(CXX) $(CFLAGS_OPT) -c $< -o $(OBJ_PATH_OPT)$*.o" >> $@

# rule for automatically generating DEBUG object file rules (.o files) using the compiler for src files
$(DBG_DEP_FILES) : $(DEP_PATH_DBG)%.dep: $(SRC_PATH)%.cpp
	@echo "generating DEBUG make rule and recipe for $*.o and storing it in $@ using the $(CXX) compiler."
	@echo -n "$(OBJ_PATH_DBG)" > $@
	@$(CXX) -MM $< >> $@
	@echo "	$(CXX) $(CFLAGS_DBG) -c $< -o $(OBJ_PATH_DBG)$*.o" >> $@

# including auto-generated make rules
-include $(OPT_DEP_FILES)
-include $(DBG_DEP_FILES)
