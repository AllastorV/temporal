#pragma once
#include "../core/core.h"
#include "../modules/print_lab_kernel.h"
#include "../modules/refraction_kernel.h"
#include "../modules/multi_optic_kernel.h"
#include "../modules/lens_character_kernel.h"
#include "../modules/rain_kernel.h"
#include "../core/cutwright_host.h"
#include <vector>

namespace tp::gpu {

bool available();
void setEnabled(bool on);
bool enabled();
bool syncStream(void* stream);
bool copyImage(const ImageView& from, const ImageView& to, const Rect& r, void* stream);
bool finishDevice(const ImageView& src, const ImageView& dst, const Rect& r, float mix, void* stream);

struct TearVectorGrid {
    const float* u;
    const float* v;
    int width, height;
    int originX, originY;
    float spacing, vectorScale;
};

struct TearExposure {
    float fixedX, fixedY, bias, threshold, shutter, diagonal;
    float sampleStart, softness, highlights, amount;
    RGBA tint;
    int samples, encoding;
    bool fromMotion;
};

bool motionTear(const ImageView& linear, const ImageView& filtered, const ImageView& dst,
                Rect window, TearVectorGrid flow, TearVectorGrid nearby, TearExposure exposure);

struct DiffusionArgs {
    ImageView src,dst;
    Rect frame,region,window;
    int mode,encoding;
    float threshold,radius,length,angle,diagonal,strength,protection;
    RGBA tint;
};
bool diffusion(const DiffusionArgs&);

struct TimeShiftArgs {
    std::vector<ImageView> taps;
    std::vector<int> subTap;
    std::vector<float> subF;
    ImageView cur;
    ImageView dst;
    Rect window;
    Rect R;
    int ds = 1;
    int mode = 0;
    int combine = 0;
    bool shadow = false;
    float dx = 0, dy = 0, cx = 0, cy = 0, radial = 0;
    float lr = 0, cov = 1, norm = 1;
    int encoding = 0;
};

bool timeShift(const TimeShiftArgs& a);

struct MotionSample {
    float a, b, c, d, tx, ty;
    float logS;
};
struct MotionArgs {
    ImageView src;
    ImageView dst;
    Rect window;
    std::vector<MotionSample> samples;
    float cx = 0, cy = 0, halfDiag = 1;
    int edge = 0;
    int encoding = 0;
};
bool motionNative(const MotionArgs& a,float mix,void* stream);
bool motionResample(const MotionArgs& a);
bool refraction(const RLParams& P, const ImageView& src, const ImageView& dst, const Rect& window);
bool multiOptic(const OpticStack& stack, const ImageView& src, const ImageView& dst, const Rect& window);
bool rainDiskBlur(const ImageView& src, const Rect& R, float radius, const ImageView& dst);
struct RainGlassArgs {
    ImageView src, dst;
    Rect R, window;
    int encoding;
    RainShade shade;
    RainSplat splat;
    const float* bead;
    const float* glint;
    float sceneBlur, dropBlur, innerSigma, fogSigma;
};
bool rainGlass(const RainGlassArgs& a);
bool lensCharacter(const LensCharacterParams& lens, const ImageView& src, const ImageView& depth,
                   const ImageView& dst, const Rect& window, SampleEdge edge);
bool halationComposite(const ImageView& cur, const ImageView& low, const ImageView& dst, const Rect& window, const Rect& region, int factor, const RGBA& tint, float gain, bool only, int encoding);
bool halationAccum(const std::vector<ImageView>& layers, const std::vector<float>& weights, const Rect& R, int f, float thr, float kw, int enc, const ImageView& low);
struct PrintLabTone {
    Rect frame, region;
    float blackPoint, range, sigma;
    int encoding;
};
bool printLab(const PLParams& P, const PLCamera& C, const PrintLabTone& tone, const ImageView& src, const ImageView& dst, const Rect& window);

struct CrtParams {
    float curvature = 0, glow = 0, lines = 240, scanlines = 0, mask = 0;
    int phosphor = 0;
};
bool cutwrightCrt(const CrtParams& p, const ImageView& exchange);
struct AnalogParams {
    CrtParams crt;
    int mode = 0;
    float amount = 1, chromatic = 0, bleed = 0, jitter = 0, roll = 0, noise = 0, dropout = 0, speed = 1, bands = 12;
    int index = 0;
    long long seed = 0;
    double fps = 24;
};
bool cutwrightAnalog(const AnalogParams& p, const ImageView& exchange);
struct MoshParams {
    int mode = 0, period = 72, index = 0;
    long long seed = 0;
    float amount = 1, motion = 1, block = 0, refresh = 0, residual = 1, memory = 0.25f, threshold = 0.002f, decay = 0.96f, chromatic = 0.3f, saturation = 1;
};
bool cutwrightMosh(const MoshParams& p, const ImageView& exchange, bool reset, bool output);
bool cutwrightStar(const cw::Plane& amount, const std::vector<cw::Plane>& light, double s1, double length, std::vector<cw::Plane>& acc);
struct VhsParams {
    float jitter = 0.3f, tracking = 0.5f, chromaBlur = 0.6f, sharpen = 0.5f, saturation = 0.7f, noise = 0.35f;
    long long seed = 0;
    int index = 0;
    double fps = 24;
};
bool cutwrightVhs(const VhsParams& p, const ImageView& exchange);
bool cutwrightFlow(const cw::Plane& cur, const cw::Plane& prev, cw::Plane& fy, cw::Plane& fx);
bool planeBlur(const cw::Plane& in, float sigma, cw::Plane& out);
struct InkParams {
    float lines = 0.6f, width = 0.0025f, hatching = 0.6f, spacing = 0.012f, angle = 45.f;
    float ink[3] = {0, 0, 0}, paper[3] = {1, 1, 1};
    int index = 0;
};
bool cutwrightInk(const InkParams& p, const ImageView& exchange, bool reset, bool output);

}
