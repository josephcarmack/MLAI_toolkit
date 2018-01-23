# makefile for deep-q-learner demo
# Author: Joseph Carmack

###################################################################
#                      Variables and Lists                        #
###################################################################

# clear and then reset the expected file extensions
.SUFFIXES:
.SUFFIXES: .cpp .o .dep

# compile and link step variables
SHELL = /bin/sh
TARGET_NAME = dqn_demo.app
DBG_NAME = dbg_$(TARGET_NAME)
CXX = g++
CFLAGS = -Wall -std=c++11
LFLAGS = -lSDL2 -lSDL2_image
CFLAGS_OPT = $(CFLAGS) -O3
CFLAGS_DBG = $(CFLAGS) -g -D_DEBUG

# path variables
SRC_PATH = ../../src/
OBJ_PATH_OPT = ../../src/object_files/opt/
DEP_PATH_OPT = ../../src/make_rules/opt/
OBJ_PATH_DBG = ../../src/object_files/dbg/
DEP_PATH_DBG = ../../src/make_rules/dbg/
GAME_PATH = cartPoleGame/
GAME_OBJ_PATH_OPT = $(GAME_PATH)object_files/opt/
GAME_OBJ_PATH_DBG = $(GAME_PATH)object_files/dbg/
GAME_DEP_PATH_OPT = $(GAME_PATH)make_rules/opt/
GAME_DEP_PATH_DBG = $(GAME_PATH)make_rules/dbg/

# name variables
MAIN_OBJ_OPT = optmain
MAIN_OBJ_DBG = dbgmain

# list files from the src directory that the demo depends on here
SRC_FILES = \
			error.cpp \
			json.cpp \
			matrix.cpp \
			neuralnet.cpp \
			rand.cpp \
			string.cpp \
			vec.cpp \

# list of cart pole game files
GAME_FILES = \
			 Controller.cpp \
			 Model.cpp \
			 View.cpp \

# build some lists of files for use in auto-generated Make rules
# use the 'printLists' rules to see their values

# lists for optimized src object files
TEMP_OBJ_OPT = $(SRC_FILES:%.cpp=%.o)
OBJ_OPT = $(TEMP_OBJ_OPT:%=$(OBJ_PATH_OPT)%)
TEMP_OPT_DEP = $(SRC_FILES:.cpp=.dep)
OPT_DEP_FILES = $(TEMP_OPT_DEP:%=$(DEP_PATH_OPT)%)

# lists for optimized cart pole game object files
TEMP_GAME_OBJ_OPT = $(GAME_FILES:%.cpp=%.o)
GAME_OBJ_OPT = $(TEMP_GAME_OBJ_OPT:%=$(GAME_OBJ_PATH_OPT)%)
TEMP_GAME_OPT_DEP = $(GAME_FILES:.cpp=.dep)
OPT_GAME_DEP_FILES = $(TEMP_GAME_OPT_DEP:%=$(GAME_DEP_PATH_OPT)%)

# lists for debug src object files
TEMP_OBJ_DBG = $(SRC_FILES:.cpp=.o)
OBJ_DBG = $(TEMP_OBJ_DBG:%=$(OBJ_PATH_DBG)%)
TEMP_DBG_DEP = $(SRC_FILES:.cpp=.dep)
DBG_DEP_FILES = $(TEMP_DBG_DEP:%=$(DEP_PATH_DBG)%)

# lists for debug cart pole game object files
TEMP_GAME_OBJ_DBG = $(GAME_FILES:%.cpp=%.o)
GAME_OBJ_DBG = $(TEMP_GAME_OBJ_DBG:%=$(GAME_OBJ_PATH_DBG)%)
TEMP_GAME_DBG_DEP = $(GAME_FILES:.cpp=.dep)
DBG_GAME_DEP_FILES = $(TEMP_GAME_DBG_DEP:%=$(GAME_DEP_PATH_DBG)%)




###################################################################
#                      Core Makefile Rules                        #
###################################################################




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
	@echo "    make fclean ------ (full clean of demo and src files)"
	@echo "    make usage"
	@echo "    or just 'make' --- (show these instructions again)"
	@echo "    ***********************************************************"
	@echo

# rule for building the optimized executable
opt : $(MAIN_OBJ_OPT).o $(OBJ_OPT) $(GAME_OBJ_OPT)
	@echo
	@echo "            **************************************"
	@echo "                LINKING OPTIMIZED OBJECT FILES"
	@echo "            **************************************"
	@echo
	$(CXX) -o $(TARGET_NAME) $(OBJ_OPT) $(GAME_OBJ_OPT) $(MAIN_OBJ_OPT).o $(LFLAGS)
	@echo
	@echo

# rule for building the debug executable
dbg : $(MAIN_OBJ_DBG).o $(OBJ_DBG) $(GAME_OBJ_DBG)
	@echo
	@echo "            **************************************"
	@echo "                LINKING OPTIMIZED OBJECT FILES"
	@echo "            **************************************"
	@echo
	$(CXX) -o $(DBG_NAME) $(OBJ_DBG) $(GAME_OBJ_DBG) $(MAIN_OBJ_DBG).o $(LFLAGS)
	@echo
	@echo

# clean up demo files
clean : 
	rm -f $(MAIN_OBJ_OPT).o $(MAIN_OBJ_OPT).dep $(MAIN_OBJ_DBG).o \
		$(MAIN_OBJ_DBG).dep $(TARGET_NAME) $(DBG_NAME) $(GAME_OBJ_OPT) \
		$(GAME_OBJ_DBG) $(OPT_GAME_DEP_FILES) $(DBG_GAME_DEP_FILES)

# clean up demo files and delete src object files 
fclean: 
	rm -f $(MAIN_OBJ_OPT).o $(MAIN_OBJ_OPT).dep $(MAIN_OBJ_DBG).o \
		$(MAIN_OBJ_DBG).dep $(TARGET_NAME) $(DBG_NAME) $(GAME_OBJ_OPT) \
		$(GAME_OBJ_DBG) $(OPT_GAME_DEP_FILES) $(DBG_GAME_DEP_FILES) \
		$(OBJ_OPT) $(OBJ_DBG)

# rule for printing values stored in list variables
# (for better understanding how the makefile works)
printLists:
	@echo
	@echo "lists for optimized src object files:"
	@echo "TEMP_OBJ_OPT = $(TEMP_OBJ_OPT)"
	@echo "OBJ_OPT = $(OBJ_OPT)"
	@echo "TEMP_OPT_DEP = $(TEMP_OPT_DEP)"
	@echo "OPT_DEP_FILES = $(OPT_DEP_FILES)"
	@echo
	@echo "lists for optimized game object files:"
	@echo "TEMP_GAME_OBJ_OPT = $(TEMP_GAME_OBJ_OPT)"
	@echo "GAME_OBJ_OPT = $(GAME_OBJ_OPT)"
	@echo "TEMP_GAME_OPT_DEP = $(TEMP_GAME_OPT_DEP)"
	@echo "OPT_GAME_DEP_FILES = $(OPT_GAME_DEP_FILES)"
	@echo
	@echo "lists for debug src object files:"
	@echo "TEMP_OBJ_DBG = $(TEMP_OBJ_DBG)"
	@echo "OBJ_DBG = $(OBJ_DBG)"
	@echo "TEMP_DBG_DEP = $(TEMP_DBG_DEP)"
	@echo "DBG_DEP_FILES = $(DBG_DEP_FILES)"
	@echo
	@echo "lists for debug game object files:"
	@echo "TEMP_GAME_OBJ_DBG = $(TEMP_GAME_OBJ_DBG)"
	@echo "GAME_OBJ_DBG = $(GAME_OBJ_DBG)"
	@echo "TEMP_GAME_DBG_DEP = $(TEMP_GAME_DBG_DEP)"
	@echo "DBG_GAME_DEP_FILES = $(DBG_GAME_DEP_FILES)"
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

# including auto-generated make rules for src files
-include $(OPT_DEP_FILES)
-include $(DBG_DEP_FILES)

# rule for automatically generating OPTIMIZED object file rules (.o files) using the compiler for game files
$(OPT_GAME_DEP_FILES) : $(GAME_DEP_PATH_OPT)%.dep: $(GAME_PATH)%.cpp
	@echo "generating OPTIMIZED make rule and recipe for $*.o and storing it in $@ using the $(CXX) compiler."
	@echo -n "$(GAME_OBJ_PATH_OPT)" > $@
	@$(CXX) -MM $< >> $@
	@echo "	$(CXX) $(CFLAGS_OPT) -c $< -o $(GAME_OBJ_PATH_OPT)$*.o" >> $@

# rule for automatically generating DEBUG object file rules (.o files) using the compiler for game files
$(DBG_GAME_DEP_FILES) : $(GAME_DEP_PATH_DBG)%.dep: $(GAME_PATH)%.cpp
	@echo "generating DEBUG make rule and recipe for $*.o and storing it in $@ using the $(CXX) compiler."
	@echo -n "$(GAME_OBJ_PATH_DBG)" > $@
	@$(CXX) -MM $< >> $@
	@echo "	$(CXX) $(CFLAGS_DBG) -c $< -o $(GAME_OBJ_PATH_DBG)$*.o" >> $@

# including auto-generated make rules for src files
-include $(OPT_GAME_DEP_FILES)
-include $(DBG_GAME_DEP_FILES)
