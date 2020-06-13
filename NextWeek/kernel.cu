#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 10; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(unsigned char* fb, int max_x, int max_y, int pitch, int ns, camera** cam, hitable** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_y + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    col /= (float)ns;
    rand_state[pixel_index] = local_rand_state;
    float* pixel = (float*)(fb + j * pitch) + 4 * i;
    pixel[0] = sqrt(col[0]);
    pixel[1] = sqrt(col[1]);
    pixel[2] = sqrt(col[2]);
    pixel[3] = 1;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
            new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        //for (int a = -11; a < 11; a++) {
        //    for (int b = -11; b < 11; b++) {
        //        float choose_mat = RND;
        //        vec3 center(a + RND, 0.2, b + RND);
        //        if (choose_mat < 0.8f) {
        //            d_list[i++] = new sphere(center, 0.2,
        //                new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
        //        }
        //        else if (choose_mat < 0.95f) {
        //            d_list[i++] = new sphere(center, 0.2,
        //                new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
        //        }
        //        else {
        //            d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
        //        }
        //    }
        //}
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, 4);

        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0; (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            30.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus);
    }
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera) {
    for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}

static int ns = 4;
static int tx = 8;
static int ty = 8;
hitable** d_list;
hitable** d_world;
camera** d_camera;
int num_hitables = 4;
vec3* fb;
curandState* d_rand_state;
curandState* d_rand_state2;
extern "C"
void cuda_raytracing_init(int width, int height)
{

    int num_pixels = width * height;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // allocate random state
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));
    // make our world of hitables & the camera
    
    checkCudaErrors(cudaMalloc((void**)&d_list, num_hitables * sizeof(hitable*)));  
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*))); 
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    create_world << <1, 1 >> > (d_list, d_world, d_camera, width, height, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

     //Render our buffer
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (width, height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
   
}

extern "C"
void cuda_raytracing_release()
{
    free_world << <1, 1 >> > (d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));
}

__global__ void cuda_kernel_texture_2d(unsigned char* surface, int width, int height, size_t pitch, float t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float* pixel;

    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
    if (x >= width || y >= height) return;

    // get a pointer to the pixel at (x,y)
    pixel = (float*)(surface + y * pitch) + 4 * x;

    // populate it
    // float value_x = 0.5f + 0.5f * cos(t + 10.0f * ((2.0f * x) / width - 1.0f));
    // float value_y = 0.5f + 0.5f * cos(t + 10.0f * ((2.0f * y) / height - 1.0f));
    pixel[0] = (float)x / width;
    pixel[1] = (float)y / width;
    pixel[2] = 0.5f;
    pixel[3] = 1; // alpha
}

extern "C"
void cuda_raytracing_render(void* surface, int width, int height, size_t pitch)
{
    cudaError_t error = cudaSuccess;

    dim3 Db = dim3(tx, ty);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

    render << <Dg, Db >> > ((unsigned char*)surface, width, height, pitch, ns, d_camera, d_world, d_rand_state);
    //cuda_kernel_texture_2d << <Dg, Db >> > ((unsigned char*)surface, width, height, pitch, 0.0f);
    // cuda_kernel_texture_2d << <width, height >> > ((unsigned char*)surface, width, height, pitch, t);

    error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        printf("cuda_kernel_texture_2d() failed to launch error = %d\n", error);
    }
}