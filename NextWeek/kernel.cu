#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include <math_constants.h>
//#include <helper_functions.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "aabb.h"
#include "lbvh.h"
#include "texture.h"
#include "rectangle.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        getchar();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, lbvh** world, curandState* local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    vec3 cur_sum = vec3(0.0, 0.0, 0.0);
    for (int i = 0; i < 10; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
            cur_sum += emitted * cur_attenuation;

            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation = cur_attenuation * attenuation;
                cur_ray = scattered;
            }
            else {
                return cur_sum;
            }
        }
        else {
            //没有天空光
            return vec3(0.0, 0.0, 0.0);
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1993, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int m = (i + j) / 100;
    if (m > 0) return;
    int k = (i + j) % 100;

    //if (i >= max_x) return;
    //int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1995, k, 0, &rand_state[k]);
}

__global__ void render(unsigned char* fb, int max_x, int max_y, size_t pitch, int ns, camera** cam, lbvh** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_y + i;
    int k = (i * 65535 + j) % 100;
    curandState local_rand_state = rand_state[k];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    col /= (float)ns;
        rand_state[k] = local_rand_state;
    float* pixel = (float*)(fb + j * pitch) + 4 * i;
    pixel[0] = sqrt(col[0]);
    pixel[1] = sqrt(col[1]);
    pixel[2] = sqrt(col[2]);
    pixel[3] = 1;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable** d_list, lbvh** d_world, camera** d_camera, int nx, int ny, curandState* rand_state, unsigned char * d_pixels, int image_width, int image_height) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        perlin* noise = new perlin();
        noise->init(&local_rand_state);

        material* red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
        material* white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
        material* green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
        material* light = new diffuse_light(new constant_texture(vec3(15, 15, 15)));

        texture_base* pertext = new noise_texture(noise);
        texture_base* imagetxt = new image_texture(d_pixels, image_width, image_height);
        d_list[0] = new yz_rect(0, 555, 0, 555, 555, green);
        d_list[1] = new yz_rect(0, 555, 0, 555, 0, red);
        d_list[2] = new xz_rect(213, 343, 227, 332, 554, light);
        d_list[3] = new xz_rect(0, 555, 0, 555, 555, white);
        d_list[4] = new xz_rect(0, 555, 0, 555, 0, white);
        d_list[5] = new xy_rect(0, 555, 0, 555, 550, white);
        *rand_state = local_rand_state;
        *d_world = new lbvh(d_list, 6);

        vec3 lookfrom(278, 278, -800);
        vec3 lookat(278, 278, 0);
        float dist_to_focus = 10.0;
        float aperture = 0.0;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            40.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus,
            0.0f, 
            1.0f);
    }
}

__global__ void free_world(hitable** d_list, lbvh** d_world, camera** d_camera) {
    for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}


__global__ void cal_aabb(lbvh** d_world) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < (*d_world)->size()) {
        (*d_world)->cal_aabb(id);
    }
}

__global__ void radix_sort(lbvh** d_world) {
    int id = threadIdx.x;
    if (id < BUCKET_SIZE) {
        (*d_world)->radix_sort(id);
    }
}

__global__ void cal_bvh(lbvh** d_world) {
    int id = threadIdx.x * blockDim.y + threadIdx.y;
    if (id < (*d_world)->size() - 1) {
        (*d_world)->cal_hierarchy(id);
    }
}

static int ns = 4;
static int tx = 8;
static int ty = 8;
static hitable** d_list;
static lbvh** d_world;
static camera** d_camera;
static int num_hitables = 4;
static vec3* fb;
static curandState* d_rand_state;
static curandState* d_rand_state2;
static unsigned char* d_pixels;
static int image_width = 0;
static int image_height = 0;

extern "C"
void cuda_init_texture(int width, int height, unsigned char* pixels)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
    unsigned char** d;
    size_t size = width * height * 4 * sizeof(unsigned char);
    cudaMallocManaged(&d_pixels, size);
    cudaMemcpy(d_pixels, pixels, size, cudaMemcpyHostToDevice);
    image_width = width;
    image_height = height;
}

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

    rand_init << <1, 1 >> > (d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    create_world << <1, 1 >> > (d_list, d_world, d_camera, width, height, d_rand_state2, d_pixels, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //求每个物体的AABB及Morton code
    cal_aabb<<<1, 512>>>(d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    radix_sort << < 1, BUCKET_SIZE >> > (d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cal_bvh << < 1, 512 >> > (d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    printf("INIT SUCCESS\n");
     //Render our buffer
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (width, height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
   
    printf("RANDOM INITED SUCCESS\n");
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

extern "C"
void cuda_raytracing_render(void* surface, int width, int height, size_t pitch)
{
    cudaError_t error = cudaSuccess;

    dim3 Db = dim3(tx, ty);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

    render << <Dg, Db >> > ((unsigned char*)surface, width, height, pitch, ns, d_camera, d_world, d_rand_state);

    error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        printf("cuda_kernel_texture_2d() failed to launch error = %d\n", error);
    }
}