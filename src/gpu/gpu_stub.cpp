#include "gpu.h"
#include "../core/flow.h"

namespace tp::gpu {

bool available() { return false; }
void setEnabled(bool) {}
bool enabled() { return false; }
bool syncStream(void*) { return false; }
bool copyImage(const ImageView&, const ImageView&, const Rect&, void*) { return false; }
bool finishDevice(const ImageView&, const ImageView&, const Rect&, float, void*) { return false; }
bool solveFlow(const std::vector<GrayImage>&, const std::vector<GrayImage>&, int, FlowField&) { return false; }
bool motionTear(const ImageView&, const ImageView&, const ImageView&, Rect, TearVectorGrid,
                TearVectorGrid, TearExposure) { return false; }
bool diffusion(const DiffusionArgs&) { return false; }
bool rainDiskBlur(const ImageView&, const Rect&, float, const ImageView&) { return false; }
bool rainGlass(const RainGlassArgs&) { return false; }
bool timeShift(const TimeShiftArgs&) { return false; }
bool motionNative(const MotionArgs&,float,void*) { return false; }
bool motionResample(const MotionArgs&) { return false; }
bool refraction(const RLParams&, const ImageView&, const ImageView&, const Rect&) { return false; }
bool multiOptic(const OpticStack&, const ImageView&, const ImageView&, const Rect&) { return false; }
bool lensCharacter(const LensCharacterParams&, const ImageView&, const ImageView&, const ImageView&,
                   const Rect&, SampleEdge) { return false; }
bool halationComposite(const ImageView& cur, const ImageView& low, const ImageView& dst, const Rect& window, const Rect& region, int factor, const RGBA& tint, float gain, bool only, int encoding) { return false; }
bool halationAccum(const std::vector<ImageView>&, const std::vector<float>&, const Rect&, int, float, float, int, const ImageView&) { return false; }
bool printLab(const PLParams&, const PLCamera&, const PrintLabTone&, const ImageView&, const ImageView&, const Rect&) { return false; }

bool cutwrightCrt(const CrtParams&, const ImageView&) { return false; }
bool cutwrightAnalog(const AnalogParams&, const ImageView&) { return false; }
bool cutwrightMosh(const MoshParams&, const ImageView&, bool, bool) { return false; }
bool cutwrightStar(const cw::Plane&, const std::vector<cw::Plane>&, double, double, std::vector<cw::Plane>&) { return false; }
bool cutwrightVhs(const VhsParams&, const ImageView&) { return false; }
bool cutwrightFlow(const cw::Plane&, const cw::Plane&, cw::Plane&, cw::Plane&) { return false; }
bool planeBlur(const cw::Plane&, float, cw::Plane&) { return false; }
bool cutwrightInk(const InkParams&, const ImageView&, bool, bool) { return false; }

}
