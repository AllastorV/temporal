#include "harness.h"
#include "gpu/gpu.h"
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>
using namespace tpt;
using Timer = std::chrono::steady_clock;

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr,"Usage: tp_preview_bench report.csv frames default|all|full|full-all [effect] [rgb24-input] [divisor] [id=value,...] [out.ppm]\n"); return 2; }
    std::ofstream report(argv[1]);
    const int frames = std::stoi(argv[2]);
    const std::string scope = argv[3], filter = argc > 4 ? argv[4] : "";
    const bool sequence = argc > 8 && std::string(argv[8]).find("%04d") != std::string::npos;
    if (frames < 1 || frames > (sequence ? 1800 : 43) || !report || (scope!="default" && scope!="all" && scope!="full" && scope!="full-all")) return 2;
    std::ifstream raw(argc > 5 ? argv[5] : "preview_1080.rgb", std::ios::binary);
    const int div = argc > 6 ? std::stoi(argv[6]) : 1;
    if (div < 1 || 1920 % div || 1080 % div) return 2;
    const Rect bounds{0, 0, 1920 / div, 1080 / div};
    std::vector<unsigned char> buffer(1920*1080*3);
    std::vector<OwnedImage> clip;
    while(raw.read(reinterpret_cast<char*>(buffer.data()),buffer.size())) {
        OwnedImage image(bounds);
        parallelRows(0,bounds.h(),[&](int y) {
            for(int x=0;x<bounds.w();++x) {
                float sum[3]{};
                for(int dy=0;dy<div;++dy) for(int dx=0;dx<div;++dx) {
                    const auto* rgb=buffer.data()+3*(1920*(1079-(y*div+dy))+x*div+dx);
                    for(int c=0;c<3;++c) sum[c]+=rgb[c];
                }
                const float k=1.f/(255.f*div*div);
                image.view.store(x,y,{sum[0]*k,sum[1]*k,sum[2]*k,1});
            }
        });
        clip.push_back(std::move(image));
    }
    if (clip.size()!=84 || !gpu::available()) return 3;
    OwnedImage output(bounds);
    report<<"effect,variant,cold_ms,median_ms,p95_ms,max_ms,missed,frames\n";
    for(const auto* m: allModules()) {
        if(!filter.empty() && m->id!=filter) continue;
        auto base=defaultValues(*m);
        if(argc > 7) {
            std::stringstream list(argv[7]);
            for(std::string item; std::getline(list,item,',');) {
                const auto eq=item.find('=');
                if(eq==std::string::npos || !base.v.count(item.substr(0,eq))) { std::fprintf(stderr,"Unknown param: %s\n",item.c_str()); return 2; }
                base.v[item.substr(0,eq)][0]=std::stod(item.substr(eq+1));
            }
        }
        const ParamSpec* choice=nullptr;
        for(const auto& p:m->params) if(p.id=="mode" || p.id=="preset" || p.id=="look") {choice=&p;break;}
        if(!choice) for(const auto& p:m->params) if(p.id=="map_source" || p.id=="source" || p.id=="blend") {if(p.type==PType::Choice) {choice=&p;break;}}
        int count=((scope=="all" || scope=="full-all") && choice)?int(choice->options.size()):1;
        for(int v=0;v<count;++v) {
            auto p=base;
            if(count>1) p.v[choice->id][0]=v;
            const std::string label=choice?choice->options[p.i(choice->id)]: "Default";
            RenderContext ctx;
            ctx.params=&p;ctx.fps=30;ctx.first=0;ctx.last=83;ctx.scale=1.0;
            ctx.frame=bounds;ctx.window=bounds;ctx.dst=output.view;ctx.preview=scope=="default" || scope=="all";
            if(p.v.count("edge")) ctx.edge=static_cast<EdgeMode>(p.i("edge"));
            if(p.v.count("encoding")) ctx.encoding=static_cast<Encoding>(p.i("encoding"));
            ctx.fetchRaw=[&](double t){
                const long k=std::lround(t)%166;
                return clip.at(size_t(k<84?k:166-k)).view;
            };
            const auto writePpm=[&](const std::string& path) {
                std::ofstream ppm(path, std::ios::binary);
                ppm<<"P6\n"<<bounds.w()<<" "<<bounds.h()<<"\n255\n";
                for(int y=bounds.h()-1;y>=0;--y) for(int x=0;x<bounds.w();++x) {
                    const auto c=output.view.load(x,y);
                    for(float f:{c.r,c.g,c.b}) ppm.put(static_cast<char>(std::lround(255.f*std::clamp(f,0.f,1.f))));
                }
            };
            std::vector<double> samples; double cold=0;
            for(int n=-1;n<frames;++n) {
                ctx.time=32+n;
                if (m->id=="double_exposure") ctx.secondSource=clip[size_t(ctx.time)+9].view;
                const auto start=Timer::now();
                runModule(*m,ctx);
                const double ms=std::chrono::duration<double,std::milli>(Timer::now()-start).count();
                if(n<0) cold=ms; else samples.push_back(ms);
                if(sequence && n>=0) {
                    char path[1024];
                    std::snprintf(path,sizeof path,argv[8],n);
                    writePpm(path);
                }
            }
            std::sort(samples.begin(),samples.end());
            const auto missed=std::count_if(samples.begin(),samples.end(),[](double ms){return ms>1000./30;});
            report<<m->id<<",\""<<label<<"\","<<std::fixed<<std::setprecision(4)<<cold<<","
                  <<samples[frames/2]<<","<<samples[size_t(std::ceil(frames*.95))-1]<<","<<samples.back()
                  <<","<<missed<<","<<frames<<std::endl;
            if(argc > 8 && !sequence) writePpm(argv[8]);
        }
    }
}
