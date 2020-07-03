#ifndef LBVHH
#define LBVHH

#include "hitable.h"

class hitable_node {
public:
    hitable* hit;
    aabb box;
    int morton_code;
};

class lbvh_node {
public:
    aabb bounds;
    int left;
    int right;
    bool built;
    bool left_leaf;
    bool right_leaf;
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
    lbvh_node* bvh;
    int list_size;
    aabb bound_all;
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
        list[index].hit->bounding_box(0, 1, list[index].box);
        aabb* p = &(list[index].box);

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
        vec3 center = (list[index].box.min() + list[index].box.max()) / 2;
        //printf("%f %f %f \n", center.x(), center.y(), center.z());
        vec3 offset = bound_all.offset(center);
        //printf("%f %f %f \n", offset.x(), offset.y(), offset.z());
        constexpr int mortonBits = 10;
        constexpr int mortonScale = 1 << mortonBits;
        list[index].morton_code = EncodeMorton3(mortonScale * offset);
        //printf("%d\n", list[index].morton_code);
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

        //拷贝到新的list,排序完成
        if (index == 0) {
            hitable_node* old_list = list;
            list = (hitable_node*)malloc(sizeof(hitable_node) * list_size);
            int* result = (PASSES & 1) ? radix_sort_temp : radix_sort_list;
            for (int i = 0; i < list_size; i++) {
                list[i] = hitable_node(old_list[result[i]]);
                //printf("%d\n", list[sorted_nodes[i]].morton_code);
                printf("%x\n", list[i].morton_code);
            }
            free(radix_sort_list);
            free(radix_sort_temp);
            free(radix_prefix_sum);
            free(old_list);
        }
        __syncthreads();
        
        //__syncthreads();
        //if (index == 0) {
        //    //free(radix_sort_list);
        //    //free(radix_sort_temp);
        //    //sorted_nodes = (PASSES & 1) ? radix_sort_temp : radix_sort_list;
        //    //free ((PASSES & 1) ? radix_sort_list : radix_sort_temp);
        //    //free(radix_prefix_sum);
        //    for (int i = 0; i < list_size; i++) {
        //        printf("%d\n", list[i].morton_code);
        //    }
        //}
    }

    //最长公共前缀位数
    __device__ int delta(int i, int j) {
        if (j < 0 || j >= list_size) {
            return -1;
        }
        else {
            int a = list[i].morton_code;
            int b = list[i].morton_code;
            if (a == b) {
                return 32 + __clz(i ^ j);
            }
            else {
                return __clz(a ^ b);
            }

        }
    }
    __device__ inline int min(int i, int j) {
        return i < j ? i : j;
    }
    __device__ inline int max(int i, int j) {
        return i < j ? j : i;
    }
    // https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
    // https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2012/11/karras2012hpg_paper.pdf
    //平行化生成BVH, 对于每一个内部节点
    __device__ void cal_hierarchy(int i) {
        if (i == 0) {
            bvh = (lbvh_node*)malloc(sizeof(lbvh_node) * (list_size - 1));
        }
        __syncthreads();
        int d = (delta(i, i + 1) - delta(i, i - 1)) >= 0 ? 1 : -1;
        int delta_min = delta(i, i - d);
        int l_max = 2;
        while (delta(i, i + l_max * d) > delta_min) {
            l_max = l_max * 2;
        }
        int l = 0;
        for (int t = l_max / 2; t != 0; t = t / 2) {
            if (delta(i, i + (l + t) * d) > delta_min) {
                l = l + t;
            }
        }
        int j = i + l * d;
        int delta_node = delta(i, j);
        int s = 0;
        for (int t = l / 2; t != 0; t /= 2) {
            if (delta(i, i + (s + t) * d) > delta_node) {
                s = s + t;
            }
        }
        int gamma = i + s * d + min(d, 0);
        //printf("%d %d %d\n", i, j, gamma);
        bvh[i].left = gamma;
        bvh[i].left_leaf = (min(i, j) == gamma);
        bvh[i].right = gamma + 1;
        bvh[i].right_leaf = (max(i, j) == (gamma + 1));
        bvh[i].built = false;
        __syncthreads();
        if (i == 0) {
            for (int k = 0; k < list_size - 1; k++) {
                printf("%d = %d - %d ( %d %d )\n", k, bvh[k].left,bvh[k].right, bvh[k].left_leaf, bvh[k].right_leaf);
            }
        }
        __syncthreads();
        while (!bvh[0].built)
        {
            if (!bvh[i].built) {
                if (bvh[i].left_leaf && bvh[i].right_leaf) {
                    bvh[i].bounds = surrounding_box(list[bvh[i].left].box, list[bvh[i].right].box);
                    bvh[i].built = true;
                }
                else if (bvh[i].left_leaf) {
                    // 等待...
                    if (bvh[bvh[i].right].built) {
                        bvh[i].bounds = surrounding_box(list[bvh[i].left].box, bvh[bvh[i].right].bounds);
                        bvh[i].built = true;
                    }
                }
                else if (bvh[i].right_leaf) {
                    // 等待...
                    if (bvh[bvh[i].left].built) {
                        bvh[i].bounds = surrounding_box(bvh[bvh[i].left].bounds, list[bvh[i].right].box);
                        bvh[i].built = true;
                    }
                }
                else {
                    if (bvh[bvh[i].left].built && bvh[bvh[i].right].built) {
                        bvh[i].bounds = surrounding_box(bvh[bvh[i].left].bounds, bvh[bvh[i].right].bounds);
                        bvh[i].built = true;
                    }
                }
            }
            __syncthreads();
        }
        if (i == 0) {
            for (int k = 0; k < list_size - 1; k++) {
                printf("%f %f %f ~ %f %f %f\n", bvh[k].bounds.min().x(), bvh[k].bounds.min().x(), bvh[k].bounds.min().x(),
                    bvh[k].bounds.max().x(), bvh[k].bounds.max().x(), bvh[k].bounds.max().x());
            }
        }
        __syncthreads();
    }

    __device__ bool lbvh::hit(const ray& r, float t_min, float t_max, hit_record& rec, int index, bool isleaf) const {
        if (isleaf) {
            if (list[index].hit->hit(r, t_min, t_max, rec)) {
                return true;
            }
        }
        else {
            bool hited = false;
            float closed = t_max;
            hit_record temp_rec;
            if (this->hit(r, t_min, closed, temp_rec, bvh[index].left, bvh[index].left_leaf)) {
                closed = temp_rec.t;
                hited = true;
                rec = temp_rec;
            }
            if (this->hit(r, t_min, closed, temp_rec, bvh[index].right, bvh[index].right_leaf)) {
                hited = true;
                rec = temp_rec;
            }
            return hited;
        }
        return false;
    }

    __device__ bool lbvh::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        return this->hit(r, t_min, t_max, rec, 0, false);
    }
};

#endif
