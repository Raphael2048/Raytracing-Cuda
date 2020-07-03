#ifndef TEXTUREH
#define TEXTUREH
#include "vec3.h"
class texture_base {
public:
	__device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

class constant_texture : public texture_base {

public:
	__device__ constant_texture() {}
	__device__ constant_texture(vec3 c) :color(c) {}
	__device__ vec3 value(float u, float v, const vec3& p) const override {
		return color;
	}
	vec3 color;
};

class checker_texture : public texture_base {
	texture_base* odd;
	texture_base* even;
public:
	__device__ checker_texture() {}
	__device__ checker_texture(texture_base* t0, texture_base* t1) :even(t0), odd(t1) {}
	__device__ vec3 value(float u, float v, const vec3& p) const override {
		float sines = sinf(10.0f * p.x()) * sinf(10.0f * p.y()) * sinf(10.0f * p.z());
		if (sines < 0)
			return odd->value(u, v, p);
		else
			return even->value(u, v, p);
	}
};
#endif