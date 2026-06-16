#include "direction.h"
#include <cstdint>
#include <cstdlib>

// Quantize the gradient direction into 4 buckets without using atan2().
//
// We compare |Gy| against |Gx| scaled by tan(22.5deg) and tan(67.5deg):
//   tan(22.5deg) ~= 0.414 ~= 2/5   -> threshold:  ay*5 < ax*2
//   tan(67.5deg) ~= 2.414 ~= 12/5  -> threshold:  ay*5 > ax*12
//
// If |Gy| is small relative to |Gx|  -> gradient is mostly horizontal -> 0 (0 deg)
// If |Gy| is large relative to |Gx|  -> gradient is mostly vertical   -> 2 (90 deg)
// Otherwise it's a diagonal: sign of (gx * gy) decides "/" (1) vs "\" (3)
void compute_direction(const int16_t* gx, const int16_t* gy,
                        uint8_t* direction, int width, int height) {
    int n = width * height;

    for (int i = 0; i < n; i++) {
        int32_t gxv = gx[i];
        int32_t gyv = gy[i];

        int32_t ax = std::abs(gxv);
        int32_t ay = std::abs(gyv);

        if (ay * 5 < ax * 2) {
            // |Gy| much smaller than |Gx| -> horizontal gradient (0 deg)
            direction[i] = 0;
        } else if (ay * 5 > ax * 12) {
            // |Gy| much larger than |Gx| -> vertical gradient (90 deg)
            direction[i] = 2;
        } else {
            // Diagonal region: decide between 45 deg ("/") and 135 deg ("\")
            // based on whether gx and gy have the same sign or opposite signs.
            if ((gxv >= 0 && gyv >= 0) || (gxv < 0 && gyv < 0)) {
                direction[i] = 3; // "\" 135 degrees
            } else {
                direction[i] = 1; // "/" 45 degrees
            }
        }
    }
}

