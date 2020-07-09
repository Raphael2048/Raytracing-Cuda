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

class perlin {
	vec3* ranfloat;
	int* perm_x;
	int* perm_y;
	int* perm_z;

	__device__  vec3* perlin_generate(curandState* state) {
		vec3* p = new vec3[256];
		for (int i = 0; i < 256; ++i)
			p[i] = unit_vector(vec3(2 * curand_uniform(state) - 1, 2 * curand_uniform(state) - 1, 2 * curand_uniform(state) - 1));
		return p;
	}

	// 生成乱序的数组
	__device__  int* perlin_generate_perm(curandState* state) {
		int* p = new int[256];
		for (int i = 0; i < 256; i++)
			p[i] = i;
		for (int i = 255; i > 0; i--) {
			int target = int(curand_uniform(state) * (i + 1));
			int tmp = p[i];
			p[i] = p[target];
			p[target] = tmp;
		}
		return p;
	}
public:
	__device__  void init(curandState* state) {
		ranfloat = perlin_generate(state);
		perm_x = perlin_generate_perm(state);
		perm_y = perlin_generate_perm(state);
		perm_z = perlin_generate_perm(state);
		for (int i = 0; i < 256; i++) {
			printf("%f\n", ranfloat[i]);
		}
	}

	__device__  float noise(const vec3& p) const {
		float u = p.x() - floorf(p.x());
		float v = p.y() - floorf(p.y());
		float w = p.z() - floorf(p.z());

		float uu = u * u * (3 - 2 * u);
		float vv = v * v * (3 - 2 * v);
		float ww = w * w * (3 - 2 * w);

		int i = floorf(p.x());
		int j = floorf(p.y());
		int k = floorf(p.z());
		float accum = 0;
		//三线性插值
		for(int di = 0; di < 2; di++)
			for (int dj = 0; dj < 2; dj++)
				for (int dk = 0; dk < 2; dk++)
				{
					vec3 sample = ranfloat[perm_x[(i + di) & 255] ^ perm_y[(j + dj) & 255] ^ perm_z[(k + dk) & 255]];
					float dotted = dot(sample, vec3(u - di, v - dj, w - dk));
					accum += (di == 0 ? 1.0f - uu : uu) * (dj == 0 ? 1.0f - vv : vv) * (dk == 0 ? 1.0f - ww : ww) * dotted;
				}
	}

	__device__ float turb(const vec3& p, int depth = 7) const {
		float accum = 0;
		vec3 temp_p = p;
		float weight = 1.0;
		for (int i = 0; i < depth; i++) {
			accum += weight * noise(temp_p);
			weight *= 0.5f;
			temp_p *= 2;
		}
		return fabsf(accum);
	}
};

class noise_texture :public texture_base {
	perlin* noise;
public:
	__device__ noise_texture() {}
	__device__ noise_texture(perlin* p):noise(p) {}
	__device__ vec3 value(float u, float v, const vec3& p) const override {
		//return vec3(1, 1, 1) * 0.5 * (1 + noise->noise(p * 10));
		//return vec3(1, 1, 1) * noise->turb(p * 10);
		return vec3(1, 1, 1) * 0.5 * (1 + sinf(10 * p.z() + 10 * noise->turb(p)));
	}
};
#endif