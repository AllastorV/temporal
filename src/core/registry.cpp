#include "../modules/modules.h"

namespace tp {

const std::vector<const ModuleDef*>& allModules() {
    static const std::vector<const ModuleDef*> list = [] {
        std::vector<ModuleDef (*)()> makers = {
            makeTimeShift, makeSlowShutter, makeEcho, makePrism, makeStutter,
            makePixel, makeDoubleExposure, makeTimeDisplacement, makeHalation,
            makePrintLab, makeDiffusionLab, makeCameraMotion, makeRefractionLab,
            makeProceduralMotion, makeMotionTear, makeDatamosh, makeBlobTracking, makeMultiOptic,
            makeAnalog, makeGeometric, makeCutwrightArt, makeCutwrightMosh,
            makeLensCharacter, makeRainGlass,
        };
        std::vector<const ModuleDef*> out;
        for (auto mk : makers) {
            auto* m = new ModuleDef(mk());
            addFrameworkParams(*m);
            out.push_back(m);
        }
        return out;
    }();
    return list;
}

}
