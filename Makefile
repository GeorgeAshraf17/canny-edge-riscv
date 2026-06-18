HOST_CXX = g++
RV_CXX = riscv64-linux-gnu-g++
RV_CXXFLAGS = -lm
GTEST_INC = $(HOME)/googletest-install/include
GTEST_LIB = $(HOME)/googletest-install/lib
RV_ARCH = -march=rv64gcv
HOST_FLAGS = -O2
SRC = src/gaussian.cpp src/sobel.cpp src/magnitude.cpp \
      src/direction.cpp src/image_io.cpp

.PHONY: test canny_rv run clean sweep run_sweep canny_rvv run_rvv

test:
	mkdir -p build_host
	$(HOST_CXX) $(HOST_FLAGS) -Isrc -I$(GTEST_INC) \
		tests/test_pipeline.cpp $(SRC) \
		-L$(GTEST_LIB) -lgtest -lgtest_main -lpthread \
		-o build_host/test_pipeline
	./build_host/test_pipeline

canny_rv:
	mkdir -p build_rv
	$(RV_CXX) $(RV_ARCH) $(RV_CXXFLAGS) -O2 $(SRC) src/main.cpp \
		-o build_rv/canny_rv

canny_rvv:
	mkdir -p build_rv
	$(RV_CXX) $(RV_ARCH) $(RV_CXXFLAGS) -O3 -DUSE_RVV_GAUSSIAN $(SRC) src/gaussian_rvv.cpp src/sobel_rvv.cpp src/main.cpp -o build_rv/canny_rvv

sweep:
	mkdir -p build_rv
	@for FLAG in O0 O2 O3 Os Ofast; do \
		echo ">>> Building with -$$FLAG ..."; \
		$(RV_CXX) $(RV_ARCH) $(RV_CXXFLAGS) -$$FLAG $(SRC) src/main.cpp \
			-o build_rv/canny_$$FLAG; \
		SIZE=$$(size build_rv/canny_$$FLAG | tail -1 | awk '{print $$4}'); \
		echo "  Binary size: $$SIZE bytes"; \
	done
	@echo "=== Sweep build done ==="

run:
	qemu-riscv64 -cpu rv64,v=true,vlen=256 -L /usr/riscv64-linux-gnu \
		./build_rv/canny_rv test_image.raw 256 256

run_rvv:
	qemu-riscv64 -cpu rv64,v=true,vlen=256 -L /usr/riscv64-linux-gnu \
		./build_rv/canny_rvv test_image.raw 256 256

run_sweep:
	@for FLAG in O0 O2 O3 Os Ofast; do \
		echo ""; \
		echo "========== -$$FLAG =========="; \
		qemu-riscv64 -L /usr/riscv64-linux-gnu \
			./build_rv/canny_$$FLAG test_image.raw 256 256; \
	done

clean:
	rm -f build_host/* build_rv/*
