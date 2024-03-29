#ifndef HITABLEH
#define HITABLEH

#include "ray.h"
#include "aabb.h"

class material;

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
    float u, v;
};

class hitable  {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
        __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const { return false; };
};

class flip_normals : public hitable {
    hitable* ptr;
public :
    __device__ flip_normals() {}
    __device__ flip_normals(hitable* p) : ptr(p) {}
    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
        if (ptr->hit(r, t_min, t_max, rec)) {
            rec.normal = -rec.normal;
            return true;
        }
        return false;
    }
    __device__ bool bounding_box(float t0, float t1, aabb& box) const override {
        return ptr->bounding_box(t0, t1, box);
    }
};

#endif
