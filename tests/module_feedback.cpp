#include "harness.h"
#include "../src/core/controls.h"
using namespace tpt;

TP_TEST(stutter_fast_bpm_fits_repeats_inside_each_beat) {
    const auto& m=find("stutter");
    auto clip=makeClip(1,1,50,[](int,int,int f){return RGBA{float(f),0,0,1};});
    for (int bpm : {151,180,240,300}) {
        auto p=with(defaultValues(m),{{"mode",2},{"hold",12},{"repeat",2},{"bpm",double(bpm)}});
        int repeated=0;
        for(int t=0;t<25;++t) {
            const int src=int(renderAt(m,clip,t,p,1,25).out.view.load(0,0).r);
            CHECK(src<=t);
            repeated += src<t;
        }
        CHECK(repeated>=8);
    }
}
TP_TEST(double_exposure_can_blend_two_independent_clips) {
    const auto& m=find("double_exposure");
    auto p=with(defaultValues(m),{{"balance",0.5},{"encoding",0}});
    auto a=constantClip({1,0,0,1},8,8,1), b=constantClip({0,0,1,1},8,8,1);
    OwnedImage out(a.rect);
    RenderContext c; c.params=&p; c.frame=c.window=a.rect; c.dst=out.view;
    c.secondSource=b.frames[0].view; c.fetchRaw=[&](double t){CHECK(t==0);return a.frames[0].view;};
    runModule(m,c);
    auto color=out.view.load(4,4);
    CHECK_NEAR(color.r,0.5,1e-6); CHECK_NEAR(color.b,0.5,1e-6);
}
TP_TEST(pixel_sort_custom_diagonal_preserves_samples) {
    const auto& m=find("pixel");
    auto clip=makeClip(8,8,1,[](int x,int y,int){float v=(14-x-y)/14.f;return RGBA{v,v,v,1};});
    auto p=with(defaultValues(m),{{"mode",2},{"direction",2},{"sort_angle",45},{"sort_low",0},{"sort_high",1}});
    auto out=renderAt(m,clip,0,p);
    for(int i=1;i<8;++i) CHECK(out.out.view.load(i,i).r>=out.out.view.load(i-1,i-1).r);
    CHECK_NEAR(mean(out.out.view).r,mean(clip.frames[0].view).r,1e-6);
}
TP_TEST(multi_optic_preview_tracks_the_selected_element) {
    const auto& m=find("multi_optic");auto clip=patternClip(64,64,1);
    auto p=with(defaultValues(m),{{"preview",1},{"e1_type",2},{"e1_size",0.25}});
    p.set("e1_position",0.25,0.25);
    auto out=renderAt(m,clip,0,p);
    CHECK_NEAR(out.out.view.load(16,16).r,1,1e-6);
    CHECK_NEAR(out.out.view.load(52,52).r,0,1e-6);
    CHECK(!controlEnabled("multi_optic","e3_position",p));
}
TP_TEST(camera_motion_framing_path_hits_end_point) {
    const auto& m=find("camera_motion");
    auto c=makeClip(64,48,25,[](int x,int y,int){return RGBA{x/64.f,y/48.f,0,1};});
    auto p=with(defaultValues(m),{{"mode",1},{"use_path",1},{"zoom",1},{"shutter",0},{"end_frame",20},{"start_frame",5},{"encoding",0}});
    p.set("center",0.25,0.5);p.set("end_point",0.75,0.5);
    CHECK_NEAR(renderAt(m,c,5,p).out.view.load(16,24).r,0.25,1e-6);
    CHECK_NEAR(renderAt(m,c,20,p).out.view.load(48,24).r,0.25,1e-6);
    CHECK(!controlEnabled("camera_motion","angle",p));
}

TP_TEST(interactive_preview_scales_secondary_and_preserves_tile_mix_and_final_resolution) {
    Rect frame{-41, 7, 1159, 55}, tile{17, 11, 890, 45};
    OwnedImage source(frame), second(frame), destination(frame);
    for (int y = frame.y1; y < frame.y2; ++y)
        for (int x = frame.x1; x < frame.x2; ++x) {
            source.view.store(x,y,{0.2f, 0.3f, 0.4f, 0.8f});
            second.view.store(x,y,{0.8f, 0.5f, 0.2f, 0.4f});
        }
    ModuleDef module;
    module.id = "double_exposure";
    addFrameworkParams(module);
    auto params = with(defaultValues(module),{{"mix", 0.25}});
    int width = 0;
    module.render = [&](const RenderContext& c) {
        width = c.frame.w();
        CHECK(c.source().valid() && c.frameOffset(-1).valid());
        copyRect(c.secondSource,c.dst,c.window);
        if (c.releaseRaw) c.releaseRaw(c.time - 1);
    };
    RenderContext ctx;
    ctx.params = &params;
    ctx.time = 2;
    ctx.last = 8;
    ctx.frame = frame;
    ctx.window = tile;
    ctx.dst = destination.view;
    ctx.secondSource = second.view;
    ctx.fetchRaw = [&](double) { return source.view; };
    for (bool preview : {true, false}) {
        ctx.preview = preview;
        runModule(module,ctx);
        CHECK(width == (preview ? 600 : 1200));
        CHECK_NEAR(destination.view.load(400,30).r, 0.35, 1e-6);
        CHECK_NEAR(destination.view.load(400,30).a, 0.70, 1e-6);
        CHECK_NEAR(destination.view.load(frame.x1,frame.y1).a, 0, 1e-6);
    }
}
