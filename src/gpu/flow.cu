#include "gpu_common.cuh"
#include "../core/flow.h"

namespace tp::gpu {
namespace {

struct Plane {
    float* data;
    int w, h;
    __device__ float at(int x, int y) const {
        return data[min(max(y,0),h-1)*w + min(max(x,0),w-1)];
    }
    __device__ float sample(float x, float y) const {
        const int i=__float2int_rd(x), j=__float2int_rd(y);
        const float tx=x-i, ty=y-j;
        const float a=at(i,j), b=at(i,j+1);
        const float row0=a+(at(i+1,j)-a)*tx;
        const float row1=b+(at(i+1,j+1)-b)*tx;
        return row0+(row1-row0)*ty;
    }
};

__global__ void expandVelocity(float* dst, int stride, int w, int h, const float* coarse, int cw, int ch) {
    const int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=w*h) return;
    const int i=cw ? min((k%w)/2,cw-1) + cw*min((k/w)/2,ch-1) : 0;
    dst[k]=cw ? 2*coarse[i] : 0;
    dst[stride+k]=cw ? 2*coarse[stride+i] : 0;
}

__global__ void tensorProducts(Plane a, Plane b, const float* flow, float* terms, int stride) {
    const int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=a.w*a.h) return;
    const int x=k%a.w, y=k/a.w;
    const float px=x+0.5f-flow[k], py=y+0.5f-flow[stride+k];
    const float gx=0.25f*(b.at(x+1,y)-b.at(x-1,y)+a.sample((px+1.f)-0.5f,py-0.5f)-a.sample((px-1.f)-0.5f,py-0.5f));
    const float gy=0.25f*(b.at(x,y+1)-b.at(x,y-1)+a.sample(px-0.5f,(py+1.f)-0.5f)-a.sample(px-0.5f,(py-1.f)-0.5f));
    const float error=b.data[k]-a.sample(px-0.5f,py-0.5f);
    terms[k]=gx*gx;
    terms[stride+k]=gx*gy;
    terms[2*stride+k]=gy*gy;
    terms[3*stride+k]=gx*error;
    terms[4*stride+k]=gy*error;
}

// Use the reference running sum order so confidence masks also agree near their threshold.
__global__ void boxLines(const float* src, float* dst, int stride, int w, int h,
                         int radius, bool vertical) {
    const int line=blockIdx.x*blockDim.x+threadIdx.x;
    if(line>=(vertical?w:h)) return;
    const int n=vertical?h:w, step=vertical?w:1;
    const int base=blockIdx.y*stride+(vertical?line:line*w);
    src+=base; dst+=base;
    float sum=0;
    for(int k=-radius;k<=radius;++k) sum+=src[min(max(k,0),n-1)*step];
    for(int k=0;k<n;++k) {
        dst[k*step]=sum/(2*radius+1);
        sum+=src[min(k+radius+1,n-1)*step]-src[max(k-radius,0)*step];
    }
}

__global__ void updateVelocity(float* flow, const float* terms, int stride, int count) {
    const int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=count) return;
    const float xx=terms[k]+1e-4f, xy=terms[stride+k], yy=terms[2*stride+k]+1e-4f;
    const float xt=terms[3*stride+k], yt=terms[4*stride+k];
    const float det=xx*yy-xy*xy;
    if(det>1e-12f) {
        flow[k]+=dclamp(-(yy*xt-xy*yt)/det,-2,2);
        flow[stride+k]+=dclamp(-(xx*yt-xy*xt)/det,-2,2);
    }
}

__global__ void confidence(Plane a, Plane b, const float* flow, float* keep, int stride) {
    const int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=a.w*a.h) return;
    const float u=flow[k],v=flow[stride+k];
    const float error=fabsf(b.data[k]-a.sample((k%a.w+0.5f-u)-0.5f,(k/a.w+0.5f-v)-0.5f));
    const float mag=hypotf(u,v);
    keep[k]=(error>0.04f+0.08f*fabsf(b.data[k]) || mag>0.12f*max(a.w,a.h) || !isfinite(mag)) ? 0.f : 1.f;
}

__global__ void rejectOutliers(float* flow, const float* keep, int count) {
    const int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=count) return;
    const float gain=dsmooth(0.5f,0.9f,keep[k]);
    float& v=flow[blockIdx.y*count+k];
    v=isfinite(v) ? v*gain : 0;
}
}

bool solveFlow(const std::vector<GrayImage>& previous, const std::vector<GrayImage>& current,
               int iterations, FlowField& output) {
    const int stride=previous.front().w*previous.front().h;
    const size_t bytes=size_t(stride)*sizeof(float);
    std::lock_guard<std::mutex> lock(deviceMutex());
    cudaStream_t stream{};
    if(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)!=cudaSuccess) return false;
    DeviceBuffer storage;
    bool ok=storage.alloc(16*bytes);
    if(ok) {
        auto* a=static_cast<float*>(storage.ptr);
        float* b=a+stride;
        float* velocity=b+stride;
        float* coarse=velocity+2*stride;
        float* terms=coarse+2*stride;
        float* scratch=terms+5*stride;
        int cw=0,ch=0;
        for(int level=int(previous.size())-1;level>=0 && ok;--level) {
            const int w=previous[level].w,h=previous[level].h,count=w*h,blocks=(count+127)/128;
            Plane first{a,w,h},second{b,w,h};
            ok=cudaMemcpyAsync(a,previous[level].v.data(),count*sizeof(float),cudaMemcpyHostToDevice,stream)==cudaSuccess &&
               cudaMemcpyAsync(b,current[level].v.data(),count*sizeof(float),cudaMemcpyHostToDevice,stream)==cudaSuccess;
            if(!ok) break;
            auto blur=[&](float* data,int planes,int radius) {
                boxLines<<<dim3((h+63)/64,planes),64,0,stream>>>(data,scratch,stride,w,h,radius,false);
                boxLines<<<dim3((w+63)/64,planes),64,0,stream>>>(scratch,data,stride,w,h,radius,true);
            };
            expandVelocity<<<blocks,128,0,stream>>>(velocity,stride,w,h,coarse,cw,ch);
            for(int it=0;it<iterations;++it) {
                tensorProducts<<<blocks,128,0,stream>>>(first,second,velocity,terms,stride);
                blur(terms,5,3);
                updateVelocity<<<blocks,128,0,stream>>>(velocity,terms,stride,count);
                blur(velocity,2,1);
            }
            if(level==0) {
                confidence<<<blocks,128,0,stream>>>(first,second,velocity,terms,stride);
                blur(terms,1,1);
                rejectOutliers<<<dim3(blocks,2),128,0,stream>>>(velocity,terms,count);
            }
            cw=w;ch=h;
            std::swap(coarse,velocity);
            ok=cudaGetLastError()==cudaSuccess;
        }
        if(ok) {
            output.reset(previous.front().w,previous.front().h);
            ok=cudaMemcpyAsync(output.u.data(),coarse,bytes,cudaMemcpyDeviceToHost,stream)==cudaSuccess &&
               cudaMemcpyAsync(output.v.data(),coarse+stride,bytes,cudaMemcpyDeviceToHost,stream)==cudaSuccess;
        }
    }
    if(cudaStreamSynchronize(stream)!=cudaSuccess) ok=false;
    cudaStreamDestroy(stream);
    return ok;
}
}
