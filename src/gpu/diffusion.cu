#include "gpu_common.cuh"

namespace tp::gpu {
namespace {
struct Level { DevImage image; int factor; };
struct Bank { Level gauss[3],star[2]; int count; float weights[3]; };
dim3 gridFor(DevImage img) { return dim3((img.w+15)/16,(img.h+15)/16); }
__device__ float4 plus(float4 a,float4 b){return f4add(a,b);}
__device__ float4 times(float4 a,float s){return f4mul(a,s);}
__device__ float4 tinted(float4 a,float4 t){return make_float4(a.x*t.x,a.y*t.y,a.z*t.z,a.w);}
__device__ float4 bilinearBlack(DevImage im,float x,float y){
    x-=0.5f;y-=0.5f;int i=__float2int_rd(x),j=__float2int_rd(y);
    return f4lerp(f4lerp(dAtBlack(im,i,j),dAtBlack(im,i+1,j),x-i),
                  f4lerp(dAtBlack(im,i,j+1),dAtBlack(im,i+1,j+1),x-i),y-j);
}
__global__ void selectLight(DevImage src, DevImage selected, float threshold,int enc,bool all) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=selected.w || j>=selected.h) return;
    float4 v=dDecode(dAtBlack(src,selected.x1+i,selected.y1+j),enc);
    if(!all){
        v=make_float4(fmaxf(0,v.x),fmaxf(0,v.y),fmaxf(0,v.z),0);
        float l=dluma(v),w=threshold*0.5f;
        float excess=fmaxf(0,l-threshold);
        if(w>1e-6f && l>threshold-w && l<threshold+w) {float d=l-threshold+w;excess=d*d/(4*w);}
        v=times(v,l>1e-8f?excess/l:0.f);
    }
    selected.data[size_t(j)*selected.w+i]=v;
}
__global__ void fogMeans(DevImage im,double* rows){
    int y=blockIdx.x*blockDim.x+threadIdx.x;if(y>=im.h)return;
    double total=0;
    for(int x=0;x<im.w;++x) total+=fmaxf(0.f,dluma(im.data[size_t(y)*im.w+x]));
    rows[y]=total;
}
__global__ void fogMean(double* rows,int w,int h){
    double sum=0;for(int y=0;y<h;++y)sum+=rows[y]; rows[0]=fmin(1e4,sum/(double(w)*h));
}
__global__ void downsample(DevImage src,DevImage out,int factor){
    int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=out.w || j>=out.h)return;
    int x0=i*factor,y0=j*factor,x1=min(x0+factor,src.w),y1=min(y0+factor,src.h);
    float4 sum=make_float4(0,0,0,0);
    for(int y=y0;y<y1;++y)for(int x=x0;x<x1;++x)sum=plus(sum,src.data[size_t(y)*src.w+x]);
    out.data[size_t(j)*out.w+i]=times(sum,1.f/((x1-x0)*(y1-y0)));
}
__global__ void boxPass(DevImage src,DevImage out,int radius,bool vertical){
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=out.w || y>=out.h)return;
    float4 sum=make_float4(0,0,0,0);
    for(int d=-radius;d<=radius;++d)sum=plus(sum,dAtClamp(src,x+(vertical?0:d),y+(vertical?d:0)));
    out.data[size_t(y)*out.w+x]=times(sum,1.f/(2*radius+1));
}
__global__ void streakPass(DevImage src,DevImage out,int taps,float dx,float dy){
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=out.w || y>=out.h)return;
    float4 sum=make_float4(0,0,0,0);float weight=0;
    for(int k=-taps;k<=taps;++k){
        float w=expf(-3.f*abs(k)/taps);weight+=w;
        sum=plus(sum,times(bilinearBlack(src,x+0.5f+dx*k,y+0.5f+dy*k),w));
    }
    out.data[size_t(y)*out.w+x]=times(sum,1.f/weight);
}
__device__ float4 sampleLevel(Level l,float x,float y,bool black=false){
    x/=l.factor;y/=l.factor;
    return black?bilinearBlack(l.image,x,y):dBilinearClamp(l.image,x,y);
}
__global__ void compose(DevImage src,DevImage out,Rect frame,Bank bank,int mode,int enc,float k,float sp,float4 tint,const double* mean) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=out.w||j>=out.h)return;
    int x=out.x1+i,y=out.y1+j;
    float px=x-src.x1+0.5f,py=y-src.y1+0.5f;
    float4 lin=dDecode(dAtBlack(src,x,y),enc),s=make_float4(0,0,0,0),o{};
    for(int n=0;n<bank.count;++n)s=plus(s,times(sampleLevel(bank.gauss[n],px,py),bank.weights[n]));
    switch(mode) {
    case 0:o=f4lerp(lin,tinted(s,tint),k);break;
    case 1:o=plus(lin,times(tinted(s,tint),k*(1-sp+sp*dsmooth(0,0.18f,dluma(lin)))));o.w=lin.w;break;
    case 2:case 8:{
        float4 st=plus(sampleLevel(bank.star[0],px,py,true),sampleLevel(bank.star[1],px,py,true));
        float4 h=times(tinted(plus(times(st,0.6f),times(s,0.4f)),tint),k);
        if(mode==8){h.y*=0.72f;h.z*=0.42f;}o=plus(lin,h);o.w=lin.w;break;}
    case 3:{
        float m=float(*mean);float4 veil=make_float4(tint.x*m,tint.y*m,tint.z*m,lin.w);
        o=plus(times(lin,1-k*0.6f),times(plus(times(s,0.7f),times(veil,0.3f)),k));
        o.w=lin.w+(s.w-lin.w)*0.7f*k;break;}
    case 4:o=plus(lin,times(tinted(sampleLevel(bank.star[0],px,py,true),tint),2*k));o.w=lin.w;break;
    case 5:{float l=dclamp(dluma(s),0,1);float4 pearl=make_float4(0.95f+0.1f*l,0.98f+0.02f*l,1.06f-0.12f*l,1);
        o=f4lerp(lin,tinted(tinted(s,pearl),tint),k);o.w=lin.w;break;}
    case 6:o=plus(lin,times(tinted(s,tint),1.5f*k));o.w=lin.w;break;
    case 7:{float l=dluma(lin);o=f4lerp(lin,s,k*dsmooth(0.02f,0.15f,l)*(1-dsmooth(0.6f,1.2f,l)));o.w=lin.w;break;}
    case 9:{
        float nx=2*(x+0.5f-frame.x1)/(frame.x2-frame.x1)-1,ny=2*(y+0.5f-frame.y1)/(frame.y2-frame.y1)-1;
        float mask=dsmooth(0.25f,1.1f,hypotf(nx,ny));
        float4 smear=times(plus(sampleLevel(bank.star[0],px,py),s),0.5f);
        o=plus(f4lerp(lin,tinted(smear,tint),dclamp(mask*k*1.6f,0,1)),times(tinted(s,tint),0.15f*k*mask));o.w=lin.w;break;}
    }
    out.data[size_t(j)*out.w+i]=dEncode(o,enc);
}
}

bool diffusion(const DiffusionArgs& a) {
    std::lock_guard<std::mutex> lock(deviceMutex());
    cudaStream_t stream=nullptr;
    if(cudaStreamCreate(&stream)!=cudaSuccess)return false;
    DeviceBuffer input,selected,output,means,storage[5],tmp;
    DevImage src{};
    bool ok=uploadRegion(a.src,a.region,input,src,stream);
    auto alloc=[&](DeviceBuffer& buf,int w,int h){
        if(ok) ok=buf.alloc(size_t(w)*h*sizeof(float4));return DevImage{static_cast<float4*>(buf.ptr),0,0,w,h};};
    DevImage sel=alloc(selected,src.w,src.h);sel.x1=src.x1;sel.y1=src.y1;
    DevImage out=alloc(output,a.window.w(),a.window.h());out.x1=a.window.x1;out.y1=a.window.y1;
    const int mode=a.mode; const dim3 block(16,16); Bank bank{};
    if(ok)selectLight<<<gridFor(sel),block,0,stream>>>(src,sel,a.threshold,a.encoding,mode==0||mode==3||mode==5||mode==7||mode==9);
    if(ok && mode==3) {
        ok=means.alloc(src.h*sizeof(double));
        if(ok){fogMeans<<<(src.h+127)/128,128,0,stream>>>(sel,static_cast<double*>(means.ptr));
               fogMean<<<1,1,0,stream>>>(static_cast<double*>(means.ptr),src.w,src.h);}
    }
    int slot=0;
    auto level=[&](float size,bool line,float angle) {
        int f=std::clamp(int(line?ceilf(size/32.f):floorf(size/2.5f)),1,64);
        int w=(src.w+f-1)/f,h=(src.h+f-1)/f;
        auto& buffer=storage[slot++];DevImage im=alloc(buffer,w,h);
        if(!ok)return Level{};
        downsample<<<gridFor(im),block,0,stream>>>(sel,im,f);
        if(line || size/f>0.35f) {
            DevImage scratch=alloc(tmp,w,h);if(!ok)return Level{};
            if(line) {
                int taps=std::clamp(int(ceilf(size/f)),1,32);
                streakPass<<<gridFor(im),block,0,stream>>>(im,scratch,taps,cosf(angle),-sinf(angle));
                std::swap(tmp,buffer);im.data=static_cast<float4*>(buffer.ptr);
            } else {
                float sigma=fminf(size/f,2000.f);int wl=int(floorf(sqrtf(4*sigma*sigma+1)));
                if(wl%2==0)--wl;
                int m=int(roundf((12*sigma*sigma-3*wl*wl-12*wl-9)/(-4.f*wl-4)));
                for(int pass=0;pass<3;++pass) {
                    int r=((pass<m?wl:wl+2)-1)/2;if(r<1)continue;
                    boxPass<<<gridFor(im),block,0,stream>>>(im,scratch,r,false);
                    boxPass<<<gridFor(im),block,0,stream>>>(scratch,im,r,true);
                }
            }
        }
        return Level{im,f};
    };
    if(ok) {
        if(mode==4)bank.star[0]=level(a.length,true,a.angle);
        else if(mode==2||mode==8){
            bank.count=1;bank.weights[0]=1;bank.gauss[0]=level(a.radius*0.125f,false,0);
            bank.star[0]=level(a.length,true,a.angle);bank.star[1]=level(a.length,true,a.angle+kPi*0.5f);
        } else if(mode==7){
            bank.count=1;bank.weights[0]=1;bank.gauss[0]=level(fminf(a.radius*0.3f,a.diagonal)*0.5f,false,0);
        } else {
            bank.count=3;float base=a.radius*(mode==3?3.f:mode==6?2.f:mode==9?1.5f:1.f);
            const float mult[]={1,3,9},wt[]={0.5f,0.3f,0.2f};
            for(int n=0;n<3;++n) {bank.gauss[n]=level(fminf(base*mult[n],a.diagonal)*0.5f,false,0);bank.weights[n]=wt[n];}
            if(mode==9)bank.star[0]=level(a.length*0.4f,true,a.angle);
        }
    }
    if(ok){
        compose<<<gridFor(out),block,0,stream>>>(src,out,a.frame,bank,mode,a.encoding,a.strength,a.protection,
            make_float4(a.tint.r,a.tint.g,a.tint.b,1),static_cast<double*>(means.ptr));
        ok=cudaGetLastError()==cudaSuccess && downloadRegion(out,a.dst,a.window,stream);
    }
    if(cudaStreamSynchronize(stream)!=cudaSuccess)ok=false;
    cudaStreamDestroy(stream);return ok;
}
}
