#pragma once
#include <cstdint>

template<typename PixelT, typename AccumT, typename KernelT>
void gaussian_blur(const PixelT* input, PixelT* output, int width, int height);

