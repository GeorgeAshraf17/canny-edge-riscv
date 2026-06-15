HOST_CXX = g++
RV_CXX   = riscv64-unknown-elf-g++

GTEST_INC = $(HOME)/googletest-install/include
GTEST_LIB = $(HOME)/googletest-install/lib

RV_FLAGS   = -march=rv64gcv -O2
HOST_FLAGS = -O2

.PHONY: test canny_rv run clean

test:
	mkdir -p build_host
	$(HOST_CXX) $(HOST_FLAGS) -Isrc -I$(GTEST_INC) \
		tests/test_pipeline.cpp \
		src/gaussian.cpp \
		src/sobel.cpp \
		src/magnitude.cpp \
		src/direction.cpp \
		src/image_io.cpp \
		-L$(GTEST_LIB) -lgtest -lgtest_main -lpthread \
		-o build_host/test_pipeline
	./build_host/test_pipeline
canny_rv:
	$(RV_CXX) $(RV_FLAGS) src/main.cpp -o build_rv/canny_rv

run:
	qemu-riscv64 -cpu rv64,v=true,vlen=256 ./build_rv/canny_rv

clean:
	rm -f build_host/* build_rv/*
