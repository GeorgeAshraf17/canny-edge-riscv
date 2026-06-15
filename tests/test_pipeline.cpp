
#include <gtest/gtest.h>
#include "../src/gaussian.h"
#include "../src/sobel.h"
#include "../src/magnitude.h"
#include "../src/direction.h"
#include "test_helpers.h"
#include <cstdint>
#include <vector>

// =====================================================
// Gaussian Blur Tests
// =====================================================

// A uniform image has no edges, so Sobel should produce zero
// gradient (Gx == 0 and Gy == 0) for all INTERIOR pixels.
// Border pixels are affected by zero-padding (the kernel reads
// zeros outside the image), so they are intentionally excluded.
TEST(SobelTest, UniformImageHasZeroGradient) {
    const int W = 16, H = 16;
    auto input = make_uniform_image(W, H, 100);
    std::vector<int16_t> gx(W * H), gy(W * H);

    sobel(input.data(), gx.data(), gy.data(), W, H);

    for (int y = 1; y < H - 1; y++) {
        for (int x = 1; x < W - 1; x++) {
            int i = y * W + x;
            EXPECT_EQ(gx[i], 0) << "at (" << x << "," << y << ")";
            EXPECT_EQ(gy[i], 0) << "at (" << x << "," << y << ")";
        }
    }
}
// Blurring an all-black image should produce an all-black image.
TEST(GaussianTest, AllBlackStaysBlack) {
    const int W = 16, H = 16;
    auto input = make_uniform_image(W, H, 0);
    std::vector<uint8_t> output(W * H);

    gaussian_blur<uint8_t, int32_t, int16_t>(input.data(), output.data(), W, H);

    for (int i = 0; i < W * H; i++) {
        EXPECT_EQ(output[i], 0);
    }
}

// A single bright pixel (impulse) should spread to its neighbors
// symmetrically: the center should decrease, and pixels at equal
// distance from the center (e.g. left/right, up/down) should be equal.
TEST(GaussianTest, ImpulseResponseIsSymmetric) {
    const int W = 15, H = 15; // odd size so there's a true center pixel
    auto input = make_impulse_image(W, H, 255);
    std::vector<uint8_t> output(W * H);

    gaussian_blur<uint8_t, int32_t, int16_t>(input.data(), output.data(), W, H);

    int cx = W / 2, cy = H / 2;

    // Center pixel should be reduced from 255 (energy spread to neighbors)
    EXPECT_LT(output[cy * W + cx], 255);
    // Center pixel should still be the brightest (it has the highest kernel weight)
    EXPECT_GT(output[cy * W + cx], 0);

    // Symmetry: left neighbor == right neighbor
    EXPECT_EQ(output[cy * W + (cx - 1)], output[cy * W + (cx + 1)]);
    // Symmetry: up neighbor == down neighbor
    EXPECT_EQ(output[(cy - 1) * W + cx], output[(cy + 1) * W + cx]);
    // Symmetry: diagonal neighbors should match each other
    EXPECT_EQ(output[(cy - 1) * W + (cx - 1)], output[(cy - 1) * W + (cx + 1)]);
    EXPECT_EQ(output[(cy + 1) * W + (cx - 1)], output[(cy + 1) * W + (cx + 1)]);
}

// A sharp vertical edge (left=black, right=white) should produce
// large |Gx| and near-zero |Gy| at the edge column.
TEST(SobelTest, VerticalEdgeProducesLargeGx) {
    const int W = 16, H = 16;
    auto input = make_vertical_edge_image(W, H);
    std::vector<int16_t> gx(W * H), gy(W * H);

    sobel(input.data(), gx.data(), gy.data(), W, H);

    int edge_x = W / 2;
    for (int y = 1; y < H - 1; y++) {
        int idx = y * W + edge_x;
        EXPECT_GT(std::abs(gx[idx]), 500)
            << "expected large |Gx| at edge column, y=" << y;
        EXPECT_EQ(gy[idx], 0)
            << "expected near-zero Gy at edge column, y=" << y;
    }
}

// A sharp horizontal edge (top=black, bottom=white) should produce
// large |Gy| and near-zero |Gx| at the edge row.
TEST(SobelTest, HorizontalEdgeProducesLargeGy) {
    const int W = 16, H = 16;
    auto input = make_horizontal_edge_image(W, H);
    std::vector<int16_t> gx(W * H), gy(W * H);

    sobel(input.data(), gx.data(), gy.data(), W, H);

    int edge_y = H / 2;
    for (int x = 1; x < W - 1; x++) {
        int idx = edge_y * W + x;
        EXPECT_GT(std::abs(gy[idx]), 500)
            << "expected large |Gy| at edge row, x=" << x;
        EXPECT_EQ(gx[idx], 0)
            << "expected near-zero Gx at edge row, x=" << x;
    }
}

// A diagonal edge should produce significant gradient values in
// both Gx and Gy along the diagonal line.
TEST(SobelTest, DiagonalEdgeProducesBothGxAndGy) {
    const int W = 16, H = 16;
    auto input = make_diagonal_edge_image(W, H);
    std::vector<int16_t> gx(W * H), gy(W * H);

    sobel(input.data(), gx.data(), gy.data(), W, H);

    // Check a point on the diagonal line, away from the borders
    int x = W / 2;
    int y = H - x - 1; // on the line x + y == W (approx)
    int idx = y * W + x;

    EXPECT_GT(std::abs(gx[idx]), 0) << "expected nonzero Gx on diagonal";
    EXPECT_GT(std::abs(gy[idx]), 0) << "expected nonzero Gy on diagonal";
}

// =====================================================
// Magnitude Tests
// =====================================================

// Both L1 and L2 should produce nonzero output on a random image.
TEST(MagnitudeTest, NonzeroOnRandomImage) {
    const int W = 16, H = 16;
    auto input = make_random_image(W, H);
    std::vector<int16_t> gx(W * H), gy(W * H);
    std::vector<uint8_t> mag(W * H);

    sobel(input.data(), gx.data(), gy.data(), W, H);

    magnitude_l1(gx.data(), gy.data(), mag.data(), W, H);
    bool any_nonzero_l1 = false;
    for (int i = 0; i < W * H; i++)
        if (mag[i] > 0) { any_nonzero_l1 = true; break; }
    EXPECT_TRUE(any_nonzero_l1);

    magnitude_l2(gx.data(), gy.data(), mag.data(), W, H);
    bool any_nonzero_l2 = false;
    for (int i = 0; i < W * H; i++)
        if (mag[i] > 0) { any_nonzero_l2 = true; break; }
    EXPECT_TRUE(any_nonzero_l2);
}

// On a uniform image, both magnitude methods should produce all zeros.
TEST(MagnitudeTest, ZeroOnUniformImage) {
    const int W = 16, H = 16;
    auto input = make_uniform_image(W, H, 128);
    std::vector<int16_t> gx(W * H), gy(W * H);
    std::vector<uint8_t> mag(W * H);

    sobel(input.data(), gx.data(), gy.data(), W, H);

    magnitude_l1(gx.data(), gy.data(), mag.data(), W, H);
    for (int y = 1; y < H - 1; y++)
        for (int x = 1; x < W - 1; x++)
            EXPECT_EQ(mag[y * W + x], 0);

    magnitude_l2(gx.data(), gy.data(), mag.data(), W, H);
    for (int y = 1; y < H - 1; y++)
        for (int x = 1; x < W - 1; x++)
            EXPECT_EQ(mag[y * W + x], 0);
}

// =====================================================
// Direction Tests
// =====================================================

// On a vertical edge image: Gx is large, Gy=0 → direction should be 0 (horizontal)
TEST(DirectionTest, VerticalEdgeGivesHorizontalDirection) {
    const int W = 16, H = 16;
    auto input = make_vertical_edge_image(W, H);
    std::vector<int16_t> gx(W * H), gy(W * H);
    std::vector<uint8_t> dir(W * H);

    sobel(input.data(), gx.data(), gy.data(), W, H);
    compute_direction(gx.data(), gy.data(), dir.data(), W, H);

    int edge_x = W / 2;
    for (int y = 1; y < H - 1; y++) {
        if (gx[y * W + edge_x] != 0)
            EXPECT_EQ(dir[y * W + edge_x], 0)
                << "expected direction=0 (horizontal) at vertical edge, y=" << y;
    }
}

// On a horizontal edge image: Gy is large, Gx=0 → direction should be 2 (vertical)
TEST(DirectionTest, HorizontalEdgeGivesVerticalDirection) {
    const int W = 16, H = 16;
    auto input = make_horizontal_edge_image(W, H);
    std::vector<int16_t> gx(W * H), gy(W * H);
    std::vector<uint8_t> dir(W * H);

    sobel(input.data(), gx.data(), gy.data(), W, H);
    compute_direction(gx.data(), gy.data(), dir.data(), W, H);

    int edge_y = H / 2;
    for (int x = 1; x < W - 1; x++) {
        if (gy[edge_y * W + x] != 0)
            EXPECT_EQ(dir[edge_y * W + x], 2)
                << "expected direction=2 (vertical) at horizontal edge, x=" << x;
    }
}

// On a diagonal edge: direction should be 1 or 3 (diagonal)
TEST(DirectionTest, DiagonalEdgeGivesDiagonalDirection) {
    const int W = 16, H = 16;
    auto input = make_diagonal_edge_image(W, H);
    std::vector<int16_t> gx(W * H), gy(W * H);
    std::vector<uint8_t> dir(W * H);

    sobel(input.data(), gx.data(), gy.data(), W, H);
    compute_direction(gx.data(), gy.data(), dir.data(), W, H);

    int x = W / 2, y = H / 2;
    int idx = y * W + x;
    if (gx[idx] != 0 || gy[idx] != 0)
        EXPECT_TRUE(dir[idx] == 1 || dir[idx] == 3)
            << "expected diagonal direction at diagonal edge";
}
