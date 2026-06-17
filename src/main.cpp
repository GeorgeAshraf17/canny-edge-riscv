#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <time.h>

#include "gaussian.h"
#include "sobel.h"
#include "magnitude.h"
#include "direction.h"
#include "image_io.h"

static double elapsed_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0
         + (end.tv_nsec - start.tv_nsec) / 1e6;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input.raw> <width> <height>\n", argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    int width  = atoi(argv[2]);
    int height = atoi(argv[3]);
    int n      = width * height;

    uint8_t*  input     = load_image(input_path, width, height);
    uint8_t*  blurred   = (uint8_t*)  aligned_alloc(64, n);
    int16_t*  gx        = (int16_t*)  aligned_alloc(64, n * sizeof(int16_t));
    int16_t*  gy        = (int16_t*)  aligned_alloc(64, n * sizeof(int16_t));
    uint8_t*  mag       = (uint8_t*)  aligned_alloc(64, n);
    uint8_t*  direction = (uint8_t*)  aligned_alloc(64, n);

    struct timespec t0, t1;
    const int RUNS = 100;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < RUNS; r++)
        gaussian_blur<uint8_t, int32_t, int16_t>(input, blurred, width, height);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t_gaussian = elapsed_ms(t0, t1) / RUNS;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < RUNS; r++)
        sobel(blurred, gx, gy, width, height);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t_sobel = elapsed_ms(t0, t1) / RUNS;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < RUNS; r++)
        magnitude_l1(gx, gy, mag, width, height);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t_magnitude = elapsed_ms(t0, t1) / RUNS;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < RUNS; r++)
        compute_direction(gx, gy, direction, width, height);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t_direction = elapsed_ms(t0, t1) / RUNS;

    double t_total = t_gaussian + t_sobel + t_magnitude + t_direction;

    printf("=== Canny Pipeline Timing (avg of %d runs) ===\n", RUNS);
    printf("Image size   : %d x %d\n", width, height);
    printf("─────────────────────────────────────────────\n");
    printf("Gaussian blur: %7.3f ms  (%4.1f%%)\n", t_gaussian,  t_gaussian/t_total*100);
    printf("Sobel        : %7.3f ms  (%4.1f%%)\n", t_sobel,     t_sobel/t_total*100);
    printf("Magnitude L1 : %7.3f ms  (%4.1f%%)\n", t_magnitude, t_magnitude/t_total*100);
    printf("Direction    : %7.3f ms  (%4.1f%%)\n", t_direction, t_direction/t_total*100);
    printf("─────────────────────────────────────────────\n");
    printf("Total        : %7.3f ms\n", t_total);

    save_image("out_blurred.raw",   blurred,   width, height);
    save_image("out_magnitude.raw", mag,       width, height);
    save_image("out_direction.raw", direction, width, height);
    printf("Output files saved.\n");

    free(input); free(blurred);
    free(gx);    free(gy);
    free(mag);   free(direction);
    return 0;
}

