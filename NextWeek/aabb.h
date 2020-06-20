#ifndef AABBH
#define AABBH

#include "vec3.h"
#include <algorithm>
class aabb {
	vec3 _min;
	vec3 _max;
	public:
		__device__ aabb() {}
		__device__ aabb(const vec3& a, const vec3& b) { _min = a, _max = b; }
		__device__ vec3 min() const { return _min; }
		__device__ vec3 max() const { return _max; }
		// https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
		__device__ bool hit(const ray& r, float tmin, float tmax) const {
			for (int a = 0; a < 3; a++) {
				float invD = 1.0f / r.direction()[a];
				float t0 = (_min[a] - r.origin()[a]) * invD;
				float t1 = (_max[a] - r.origin()[a]) * invD;
				if (invD < 0.0f) std::swap(t0, t1);
				tmin = t0 > tmin ? t0 : tmin;
				tmax = t1 < tmax ? t1 : tmax;
				if (tmax <= tmin) return false;
			}
			return true;
		}

		__device__ vec3 offset(const vec3& p) const {
			return (p - _min) / (_max - _min);
		}
};

__device__ aabb surrounding_box(const aabb& box0, const aabb& box1) {
	vec3 small(fminf(box0.min().x(), box1.min().x()),
		fminf(box0.min().y(), box1.min().y()),
		fminf(box0.min().z(), box1.min().z()));
	vec3 big(fmaxf(box0.max().x(), box1.max().x()),
		fmaxf(box0.max().y(), box1.max().y()),
		fmaxf(box0.max().z(), box1.max().z()));
	return aabb(small, big);
}

#endif