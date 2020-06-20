#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray
{
    public:
        __device__ ray() {}
        __device__ ray(const vec3& a, const vec3& b, float time = 0.0f) { A = a; B = b; _time = time; }
        __device__ inline vec3 origin() const       { return A; }
        __device__ inline vec3 direction() const    { return B; }
        __device__ inline float time() const { return _time; }
        __device__ inline vec3 point_at_parameter(float t) const { return A + t*B; }

        vec3 A;
        vec3 B;
        float _time;
};

#endif
