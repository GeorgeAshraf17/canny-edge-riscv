#pragma once
#include <cstdint>
#include <cstdlib>
#include <vector>

// Creates a uniform image where every pixel has the same value.
inline std::vector<uint8_t> make_uniform_image(int width, int height, uint8_t value) {
    return std::vector<uint8_t>(width * height, value);
}

// Creates an image with a sharp vertical edge:
// left half = 0 (black), right half = 255 (white).
// This produces a strong horizontal gradient (large |Gx|, near-zero |Gy|).
inline std::vector<uint8_t> make_vertical_edge_image(int width, int height) {
    std::vector<uint8_t> img(width * height, 0);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img[y * width + x] = (x < width / 2) ? 0 : 255;
        }
    }
    return img;
}

// Creates an image with a sharp horizontal edge:
// top half = 0 (black), bottom half = 255 (white).
// This produces a strong vertical gradient (large |Gy|, near-zero |Gx|).
inline std::vector<uint8_t> make_horizontal_edge_image(int width, int height) {
    std::vector<uint8_t> img(width * height, 0);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img[y * width + x] = (y < height / 2) ? 0 : 255;
        }
    }
    return img;
}

// Creates an image with a diagonal edge:
// pixels above the diagonal (x + y < width) are 0, below are 255.
// This produces significant gradient values in both Gx and Gy.
inline std::vector<uint8_t> make_diagonal_edge_image(int width, int height) {
    std::vector<uint8_t> img(width * height, 0);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img[y * width + x] = ((x + y) < width) ? 0 : 255;
        }
    }
    return img;
}

// Creates an image that is all zero except for a single bright pixel
// at the center. Used to test the impulse response of the Gaussian blur.
inline std::vector<uint8_t> make_impulse_image(int width, int height, uint8_t peak = 255) {
    std::vector<uint8_t> img(width * height, 0);
    img[(height / 2) * width + (width / 2)] = peak;
    return img;
}

// Creates an image filled with pseudo-random values in [0, 255].
// Uses a fixed seed so the test is deterministic/reproducible.
inline std::vector<uint8_t> make_random_image(int width, int height, unsigned seed = 42) {
    std::vector<uint8_t> img(width * height);
    srand(seed);
    for (auto& p : img) {
        p = static_cast<uint8_t>(rand() % 256);
    }
    return img;
}
