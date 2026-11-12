#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct BeltramiParameters {
    uint width;
    uint height;
    uint radius;
    uint reserved;
    float beta;
    float dt;
};

inline float4 gradientMetric(float4 dx, float4 dy) {
    return float4((dx.x*dx.x + dx.y*dx.y) + dx.z*dx.z,
                  (dx.x*dy.x + dx.y*dy.y) + dx.z*dy.z,
                  (dy.x*dy.x + dy.y*dy.y) + dy.z*dy.z, 0.0f);
}

inline void writeFlux(uint index, uint2 point, device const float4 *image,
                      float4 metric, constant BeltramiParameters &p,
                      device float4 *geometry, device float4 *fluxX,
                      device float4 *fluxY) {
    float4 dx = point.y ? image[index] - image[index-p.width] : float4(0.0f);
    float4 dy = point.x ? image[index] - image[index-1] : float4(0.0f);
    float g11 = p.beta + metric.x;
    float g12 = metric.y;
    float g22 = p.beta + metric.z;
    float q = pow(g11*g22 - g12*g12, -0.5f);
    geometry[index] = float4(g11, g12, g22, q);
    fluxX[index] = q * (g22*dx - g12*dy);
    fluxY[index] = q * (-g12*dx + g11*dy);
}

kernel void beltrami_metric(device const float4 *image [[buffer(0)]],
                            device float4 *metric [[buffer(1)]],
                            constant BeltramiParameters &p [[buffer(2)]],
                            uint2 point [[thread_position_in_grid]]) {
    if (point.x >= p.width || point.y >= p.height) return;
    uint index = point.y*p.width + point.x;
    float4 dx = point.y ? image[index] - image[index-p.width] : float4(0.0f);
    float4 dy = point.x ? image[index] - image[index-1] : float4(0.0f);
    metric[index] = gradientMetric(dx, dy);
}

kernel void beltrami_box_horizontal(device const float4 *metric [[buffer(0)]],
                                    device float4 *summed [[buffer(1)]],
                                    constant BeltramiParameters &p [[buffer(2)]],
                                    threadgroup float4 *tile [[threadgroup(0)]],
                                    uint2 point [[thread_position_in_grid]],
                                    uint2 local [[thread_position_in_threadgroup]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint2 size [[threads_per_threadgroup]],
                                    uint lane [[thread_index_in_threadgroup]]) {
    uint tileWidth = size.x + 2*p.radius;
    uint count = tileWidth*size.y;
    for (uint i=lane; i<count; i+=size.x*size.y) {
        int x = int(group.x*size.x + i%tileWidth) - int(p.radius);
        uint y = group.y*size.y + i/tileWidth;
        tile[i] = x>=0 && x<int(p.width) && y<p.height
                    ? metric[y*p.width+uint(x)] : float4(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (point.x >= p.width || point.y >= p.height) return;
    float4 value = float4(0.0f);
    for (uint offset=0; offset<2*p.radius+1; ++offset)
        value += tile[local.y*tileWidth + local.x + offset];
    summed[point.y*p.width+point.x] = value;
}

kernel void beltrami_flux_vertical(device const float4 *image [[buffer(0)]],
                                   device const float4 *summed [[buffer(1)]],
                                   device float4 *geometry [[buffer(2)]],
                                   device float4 *fluxX [[buffer(3)]],
                                   device float4 *fluxY [[buffer(4)]],
                                   constant BeltramiParameters &p [[buffer(5)]],
                                   threadgroup float4 *tile [[threadgroup(0)]],
                                   uint2 point [[thread_position_in_grid]],
                                   uint2 local [[thread_position_in_threadgroup]],
                                   uint2 group [[threadgroup_position_in_grid]],
                                   uint2 size [[threads_per_threadgroup]],
                                   uint lane [[thread_index_in_threadgroup]]) {
    uint count = size.x*(size.y+2*p.radius);
    for (uint i=lane; i<count; i+=size.x*size.y) {
        uint x = group.x*size.x + i%size.x;
        int y = int(group.y*size.y + i/size.x) - int(p.radius);
        tile[i] = x<p.width && y>=0 && y<int(p.height)
                    ? summed[uint(y)*p.width+x] : float4(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (point.x >= p.width || point.y >= p.height) return;
    float4 value = float4(0.0f);
    for (uint offset=0; offset<2*p.radius+1; ++offset)
        value += tile[(local.y+offset)*size.x+local.x];
    writeFlux(point.y*p.width+point.x, point, image, value, p, geometry, fluxX, fluxY);
}

kernel void beltrami_flux_local(device const float4 *image [[buffer(0)]],
                                device float4 *geometry [[buffer(1)]],
                                device float4 *fluxX [[buffer(2)]],
                                device float4 *fluxY [[buffer(3)]],
                                constant BeltramiParameters &p [[buffer(4)]],
                                uint2 point [[thread_position_in_grid]]) {
    if (point.x >= p.width || point.y >= p.height) return;
    uint index = point.y*p.width+point.x;
    float4 dx = point.y ? image[index]-image[index-p.width] : float4(0.0f);
    float4 dy = point.x ? image[index]-image[index-1] : float4(0.0f);
    writeFlux(index, point, image, gradientMetric(dx, dy), p, geometry, fluxX, fluxY);
}

kernel void beltrami_update(device const float4 *image [[buffer(0)]],
                            device const float4 *geometry [[buffer(1)]],
                            device const float4 *fluxX [[buffer(2)]],
                            device const float4 *fluxY [[buffer(3)]],
                            device float4 *output [[buffer(4)]],
                            constant BeltramiParameters &p [[buffer(5)]],
                            uint2 point [[thread_position_in_grid]]) {
    if (point.x >= p.width || point.y >= p.height) return;
    uint index = point.y*p.width+point.x;
    float4 dx = point.y+1 < p.height ? fluxX[index+p.width]-fluxX[index] : float4(0.0f);
    float4 dy = point.x+1 < p.width ? fluxY[index+1]-fluxY[index] : float4(0.0f);
    output[index] = image[index] + (p.dt*geometry[index].w)*(dx+dy);
}
