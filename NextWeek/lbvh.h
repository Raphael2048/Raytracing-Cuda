#ifndef LBVHH
#define LBVHH

#include "hitable.h"

class hitable_node {
public:
    hitable* hit;
    aabb aabb;
    int morton_code;
};

class lbvh_node {
public:
    aabb bounds;
    union {
        int index; //叶子节点
        int second_child; // 内部节点,记录第二个child节点索引
    };
    union {
        int count : 30; //内部节点, 节点数之和
        int axis : 2; //分割轴 0,1,2 XYZ
    };
};

__device__ static inline uint32_t LeftShift3(uint32_t x) {
    if (x == (1 << 10)) --x;
    x = (x | (x << 16)) & 0b00000011000000000000000011111111;
    x = (x | (x <<  8)) & 0b00000011000000001111000000001111;
    x = (x | (x <<  4)) & 0b00000011000011000011000011000011;
    x = (x | (x <<  2)) & 0b00001001001001001001001001001001;
    return x;
}

__device__ static inline uint32_t EncodeMorton3(const vec3& v) {
    return (LeftShift3(v.z()) << 2) | (LeftShift3(v.y()) << 1) | LeftShift3(v.x());
}

constexpr int BUCKET_BITS = 6;
constexpr int BUCKET_SIZE = 1 << BUCKET_BITS;
constexpr int PASSES = 30 / BUCKET_BITS;

class lbvh{
    hitable_node* list;
    int list_size;
    aabb bound_all;
    //基数排序后的索引值
    int* sorted_nodes;
    //这些是基数排序用,直接写在类里面了
    //int* radix_sort_list;
    //int* radix_sort_temp;
    //int* radix_prefix_sum;
    //int minx, miny, minz, maxx, maxy, maxz;
public:
    __device__ lbvh() {}
    __device__ lbvh(hitable** l, int n) {
        list = (hitable_node*)malloc(sizeof(hitable_node) * n);
        for (int i = 0; i < n; i++) {
            list[i] = hitable_node();
            list[i].hit = l[i];
        }
        list_size = n;

        //minx = miny = minz = 1e10;
        //maxx = maxy = maxz = -1e10;
    }

    __device__ inline int size() { return list_size; }

    //计算每个节点的aabb
    __device__ void cal_aabb(int index) {
        list[index].hit->bounding_box(0, 1, list[index].aabb);
        aabb* p = &(list[index].aabb);

        __shared__ int minx, miny, minz, maxx, maxy, maxz;
        if (index == 0) {
            minx = miny = minz = 1e10;
            maxx = maxy = maxz = -1e10;
        }
        __syncthreads();

        atomicMax(&maxx, p->max().x());
        atomicMax(&maxy, p->max().y());
        atomicMax(&maxz, p->max().z());
        atomicMin(&minx, p->min().x());
        atomicMin(&miny, p->min().y());
        atomicMin(&minz, p->min().z());
        __syncthreads();

        if (index == 0) {
            vec3 min = vec3(minx - 1, miny - 1, minz - 1);
            vec3 max = vec3(maxx - 1, maxy - 1, maxz - 1);
            bound_all = aabb(min, max);
            printf("%f %f %f \n", bound_all.min().x(), bound_all.min().y(), bound_all.min().z());
            printf("%f %f %f \n", bound_all.max().x(), bound_all.max().y(), bound_all.max().z());
        }
        __syncthreads();

        cal_morton_code(index);


    }

    __device__ void cal_morton_code(int index) {
        vec3 center = (list[index].aabb.min() + list[index].aabb.max()) / 2;
        printf("%f %f %f \n", center.x(), center.y(), center.z());
        vec3 offset = bound_all.offset(center);
        printf("%f %f %f \n", offset.x(), offset.y(), offset.z());
        constexpr int mortonBits = 10;
        constexpr int mortonScale = 1 << mortonBits;
        list[index].morton_code = EncodeMorton3(mortonScale * offset);
        printf("%d\n", list[index].morton_code);
    }

    //傻瓜版并行基数排序,考虑到几何体数量过少,不用分片分区的复杂高级算法 5次PASS完成
    __device__ void radix_sort(int index) {
        __shared__ int* radix_sort_list;
        __shared__ int* radix_sort_temp;
        __shared__ int* radix_prefix_sum;

        if (index == 0) {
            radix_sort_list = (int*)malloc(sizeof(int) * list_size);
            radix_sort_temp = (int*)malloc(sizeof(int) * list_size);
            radix_prefix_sum = (int*)malloc(sizeof(int) * BUCKET_SIZE);
            //初始化
            for (int i = 0; i < list_size; i++) {
                radix_sort_list[i] = i;
            }
        }
        __syncthreads();
        for (int i = 0; i < PASSES; i++) {
            //每次pass前先清零计数
            radix_prefix_sum[index] = 0;
            int* src = (i & 1) ? radix_sort_temp : radix_sort_list;
            int* dest = (i & 1) ? radix_sort_list : radix_sort_temp;

            //__syncthreads();
            //if (index == 0) {
            //    for (int i = 0; i < list_size; i++) {
            //        printf("%d\n", list[src[i]].morton_code);
            //    }
            //    printf("\n");
            //}
            //__syncthreads();

            int lower_bits = i * BUCKET_BITS;
            int bitMask = (1 << BUCKET_BITS) - 1;
            //printf("%d %d\n", lower_bits, bitMask);
            for (int j = 0; j < list_size; j++) {
                int code = (list[src[j]].morton_code >> lower_bits) & bitMask;
                if (code == index) {
                    radix_prefix_sum[index] ++;
                    //printf("%d %d -> %d\n", index, src[j], radix_prefix_sum[index]);
                }
            }
            __syncthreads();

            int sum = 0;
            for (int j = 0; j < index; j++) {
                sum += radix_prefix_sum[j];
            }
            __syncthreads();
            //全部计算完成后再写入
            radix_prefix_sum[index] = sum;
            __syncthreads();
            //printf("%d %d\n", index, radix_prefix_sum[index]);

            //每个thread只访问自己位置的统计值
            for (int j = 0; j < list_size; j++) {
                int code = (list[src[j]].morton_code >> lower_bits) & bitMask;
                if (code == index) {
                    //printf("%d %d -> %d\n", index, src[j], radix_prefix_sum[index]);
                    dest[radix_prefix_sum[index]++] = src[j];
                }
            }
            __syncthreads();
            //if (index == 0) {
            //    for (int j = 0; j < list_size; j++) {
            //        printf("%X\n", list[dest[j]].morton_code & ((1 << (BUCKET_BITS * (i+1))) -1));
            //    }
            //    printf("\n");
            //}
            //__syncthreads();
        }

        if (index == 0) {
            sorted_nodes = (PASSES & 1) ? radix_sort_temp : radix_sort_list;
            free ((PASSES & 1) ? radix_sort_list : radix_sort_temp);
            free(radix_prefix_sum);
            for (int i = 0; i < list_size; i++) {
                printf("%d\n", list[sorted_nodes[i]].morton_code);
            }
        }
    }

    __device__ bool lbvh::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) {
            if (list[i].hit->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }
};

#endif
