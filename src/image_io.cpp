#include "image_io.h"
#include <cstdlib>
#include <cstdio>

uint8_t* load_image(const char* path, int width, int height) {
    uint8_t* buf = (uint8_t*)aligned_alloc(64, width * height);
    FILE* f = fopen(path, "rb");
    size_t bytes_to_read = (size_t)(width * height);
    size_t bytes_read = fread(buf, 1, bytes_to_read, f);
    if (bytes_read != bytes_to_read) {
    (void)bytes_read; }
    fclose(f);
    return buf;
}

void save_image(const char* path, const uint8_t* img, int width, int height) {
    FILE* f = fopen(path, "wb");
    fwrite(img, 1, width * height, f);
    fclose(f);
}
