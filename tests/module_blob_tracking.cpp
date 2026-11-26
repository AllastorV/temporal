#include "harness.h"

using namespace tpt;

static Clip btTwoDots() {
    return makeClip(160, 96, 40, [](int x, int y, int f) {
        bool a = std::hypot(x - (20.f + f * 2.f), y - 30.f) < 6.f;
        bool b = std::hypot(x - (130.f - f * 1.5f), y - 70.f) < 8.f;
        return (a || b) ? RGBA{0.95f, 0.95f, 0.95f, 1} : RGBA{0.05f, 0.05f, 0.05f, 1};
    });
}

TP_TEST(blob_tracking_mask_matches_blobs) {
    const ModuleDef& m = find("blob_tracking");
    Clip c = btTwoDots();
    RenderResult r = renderAt(m, c, 10, with(defaultValues(m), {{"style", 5}, {"encoding", 0}, {"min_area", 0}}));
    CHECK(r.out.view.load(40, 30).r == 1.f);
    CHECK(r.out.view.load(115, 70).r == 1.f);
    CHECK(r.out.view.load(80, 50).r == 0.f);
}

TP_TEST(blob_tracking_boxes_follow_blobs_and_leave_background) {
    const ModuleDef& m = find("blob_tracking");
    Clip c = btTwoDots();
    RenderResult r = renderAt(m, c, 10, with(defaultValues(m), {{"style", 1}, {"encoding", 0}, {"min_area", 0}, {"smoothing", 0}, {"line_width", 1}}));
    double edge = 0;
    for (int y = 20; y < 40; ++y) for (int x = 28; x < 52; ++x) { RGBA o = r.out.view.load(x, y); if (o.r > 0.9f && o.b < 0.4f) edge += 1; }
    CHECK(edge > 10);
    CHECK(maxDiff(r.out.view, c.frames[10].view) > 0.1);
    RGBA bg = r.out.view.load(80, 10);
    CHECK_NEAR(bg.r, 0.05, 1e-5);
}

TP_TEST(blob_tracking_is_seek_deterministic_and_segment_bounded) {
    const ModuleDef& m = find("blob_tracking");
    Clip c = btTwoDots();
    ParamValues p = with(defaultValues(m), {{"style", 0}, {"encoding", 0}, {"min_area", 0}});
    RenderResult a = renderAt(m, c, 30, p);
    renderAt(m, c, 3, p);
    CHECK(maxDiff(a.out.view, renderAt(m, c, 30, p).out.view) == 0.0);
    CHECK(a.window.before == 24);
    int lo = 30 - a.window.before;
    for (int fr : a.fetched) CHECK(fr >= lo && fr <= 30);
}

TP_TEST(blob_tracking_nodes_links_neighbouring_blobs) {
    const ModuleDef& m = find("blob_tracking");
    Clip c = btTwoDots();
    ParamValues p = with(defaultValues(m), {{"style", 6}, {"encoding", 0}, {"min_area", 0}, {"smoothing", 0}, {"line_width", 2}, {"link_distance", 100}});
    RenderResult r = renderAt(m, c, 10, p);
    float ax = 40.f, ay = 30.f, bx = 115.f, by = 70.f;
    int hits = 0;
    for (int k = 3; k <= 7; ++k) {
        float s = k / 10.f;
        int x = static_cast<int>(ax + (bx - ax) * s), y = static_cast<int>(ay + (by - ay) * s);
        bool on = false;
        for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) { RGBA o = r.out.view.load(x + dx, y + dy); if (o.r > 0.9f && o.b < 0.4f) on = true; }
        hits += on ? 1 : 0;
    }
    CHECK(hits >= 4);
    RenderResult none = renderAt(m, c, 10, with(p, {{"connections", 0}}));
    RGBA mid = none.out.view.load(77, 50);
    CHECK_NEAR(mid.r, 0.05, 1e-5);
}

TP_TEST(blob_tracking_dark_and_motion_detect_dim_objects_and_ignore_static_highlights) {
    auto clip=makeClip(80,40,4,[](int x,int y,int frame) {
        float value=0.18f;
        if (x>=5 && x<13 && y>=10 && y<18) value=0.95f;
        if (x>=22+4*frame && x<32+4*frame && y>=12 && y<24) value=0.032f;
        return RGBA{value,value,value,1};
    });
    const auto& m=find("blob_tracking");
    auto p=with(defaultValues(m),{{"style",5},{"encoding",1},{"min_area",0},{"dark_threshold",0.08}});
    for (int mode=0;mode<=2;++mode) {
        p.set("source",mode);
        auto r=renderAt(m,clip,2,p);
        CHECK_NEAR(r.out.view.load(7,14).r,mode==0?1:0,1e-6);
        CHECK_NEAR(r.out.view.load(38,16).r,mode==0?0:1,1e-6);
        CHECK_NEAR(r.out.view.load(60,15).r,0,1e-6);
    }
}
TP_TEST(blob_tracking_curvature_bends_links_in_both_directions) {
    const auto& m=find("blob_tracking");auto clip=btTwoDots();
    auto p=with(defaultValues(m),{{"style",6},{"min_area",0},{"smoothing",0},{"encoding",0},{"link_distance",100}});
    auto straight=renderAt(m,clip,10,p);
    auto up=renderAt(m,clip,10,with(p,{{"link_curve",1}}));
    auto down=renderAt(m,clip,10,with(p,{{"link_curve",-1}}));
    CHECK(maxDiff(straight.out.view,up.out.view)>0.5);
    CHECK(maxDiff(up.out.view,down.out.view)>0.5);
}

TP_TEST(blob_tracking_inverts_only_box_interiors_and_keeps_overlay) {
    auto clip=makeClip(48,40,5,[](int x,int y,int) {
        float shade = x>=10 && x<32 && y>=8 && y<27 ? 0.8f : 0.15f;
        return RGBA{shade,shade,shade,1};
    });
    const auto& m=find("blob_tracking");
    auto off=with(defaultValues(m),{{"style",1},{"encoding",1},{"max_area",90}});
    auto on=with(off,{{"negative_boxes",1}});
    auto base=renderAt(m,clip,2,off), neg=renderAt(m,clip,2,on);
    CHECK_NEAR(neg.out.view.load(20,16).r,0.2f,1e-5);
    CHECK_NEAR(base.out.view.load(20,16).r,0.8f,1e-5);
    for (auto point : {std::pair{3,3},std::pair{10,15}}) {
        auto a=neg.out.view.load(point.first,point.second),b=base.out.view.load(point.first,point.second);
        CHECK_NEAR(a.r,b.r,1e-6); CHECK_NEAR(a.g,b.g,1e-6);
    }
}
