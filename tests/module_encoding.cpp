#include "harness.h"

using namespace tpt;

namespace {

double refDecode(float v, Encoding e) {
    switch (e) {
    case Encoding::Gamma24: return v < 0.f ? -std::pow(-v, 2.4f) : std::pow(v, 2.4f);
    case Encoding::DaVinciIntermediate: return v <= 0.02740668f ? v / 10.44426855f : std::exp2(v / 0.07329248f - 7.0f) - 0.0075f;
    case Encoding::ACEScct: return v <= 0.155251141552511f ? (v - 0.0729055341958355f) / 10.5402377416545f : std::exp2(v * 17.52f - 9.72f);
    default: return v;
    }
}

double refEncode(float v, Encoding e) {
    switch (e) {
    case Encoding::Gamma24: return v < 0.f ? -std::pow(-v, 1.f / 2.4f) : std::pow(v, 1.f / 2.4f);
    case Encoding::DaVinciIntermediate: return v <= 0.00262409f ? v * 10.44426855f : (std::log2(v + 0.0075f) + 7.0f) * 0.07329248f;
    case Encoding::ACEScct: return v <= 0.0078125f ? 10.5402377416545f * v + 0.0729055341958355f : (std::log2(v) + 9.72f) / 17.52f;
    default: return v;
    }
}

}

TP_TEST(fast_encoding_matches_std_math) {
    double worst[2][4] = {};
    for (int i = -6000; i <= 6000; ++i) {
        const float mag = std::pow(10.f, std::fabs(i) / 1000.f - 4.f);
        const float v = i < 0 ? -mag : (i == 0 ? 0.f : mag);
        for (Encoding enc : {Encoding::Gamma24, Encoding::DaVinciIntermediate, Encoding::ACEScct}) {
            const int k = static_cast<int>(enc);
            const double rd = refDecode(v, enc), re = refEncode(v, enc);
            const RGBA d = decode(RGBA{v, 0.5f * v, 0.25f, 0.7f}, enc), en = encode(RGBA{v, 0.5f * v, 0.25f, 0.7f}, enc);
            const float ds = decodeValue(v, enc), es = encodeValue(v, enc);
            const auto rel = [](double got, double want) { return std::isfinite(want) ? std::fabs(got - want) / std::max(1e-6, std::fabs(want)) : (got == want ? 0.0 : 1.0); };
            if (std::isfinite(rd) && std::fabs(rd) < 3e38) worst[0][k] = std::max({worst[0][k], rel(d.r, rd), rel(ds, rd), rel(d.g, refDecode(0.5f * v, enc))});
            if (std::isfinite(re)) worst[1][k] = std::max({worst[1][k], rel(en.r, re), rel(es, re), rel(en.g, refEncode(0.5f * v, enc))});
            CHECK(d.a == 0.7f && en.a == 0.7f);
        }
    }
    for (Encoding enc : {Encoding::Gamma24, Encoding::DaVinciIntermediate, Encoding::ACEScct}) {
        CHECK(std::isnan(decode(RGBA{std::nanf(""), 0, 0, 1}, enc).r));
        CHECK(std::isnan(encode(RGBA{0, std::nanf(""), 0, 1}, enc).g));
        CHECK(std::isinf(decode(RGBA{INFINITY, 0, 0, 1}, enc).r));
    }
    CHECK(decode(RGBA{0, 1, -1, 1}, Encoding::Gamma24).r == 0.f);
    CHECK(decode(RGBA{0, 1, -1, 1}, Encoding::Gamma24).g == 1.f);
    CHECK(decode(RGBA{0, 1, -1, 1}, Encoding::Gamma24).b == -1.f);
    for (int k = 1; k <= 3; ++k) {
        std::printf("  encoding %d rel error: decode %.2e encode %.2e\n", k, worst[0][k], worst[1][k]);
        CHECK(worst[0][k] < 2e-6);
        CHECK(worst[1][k] < 2e-6);
    }
}

TP_TEST(simd_gaussian_blur_is_bit_exact) {
    const auto refH = [](const ImageView& src, const ImageView& dst, int r) {
        const Rect& b = src.bounds;
        for (int y = b.y1; y < b.y2; ++y) {
            RGBA acc;
            const float inv = 1.f / static_cast<float>(2 * r + 1);
            for (int k = -r; k <= r; ++k) acc += src.at(b.x1 + k, y);
            for (int x = b.x1; x < b.x2; ++x) {
                dst.store(x, y, acc * inv);
                acc += src.at(x + r + 1, y) - src.at(x - r, y);
            }
        }
    };
    const auto refV = [](const ImageView& src, const ImageView& dst, int r) {
        const Rect& b = src.bounds;
        for (int x = b.x1; x < b.x2; ++x) {
            RGBA acc;
            const float inv = 1.f / static_cast<float>(2 * r + 1);
            for (int k = -r; k <= r; ++k) acc += src.at(x, b.y1 + k);
            for (int y = b.y1; y < b.y2; ++y) {
                dst.store(x, y, acc * inv);
                acc += src.at(x, y + r + 1) - src.at(x, y - r);
            }
        }
    };
    for (float sigma : {0.9f, 3.7f, 41.f}) {
        OwnedImage img(Rect{-7, 3, 90, 56});
        for (int y = 3; y < 56; ++y)
            for (int x = -7; x < 90; ++x) img.view.store(x, y, RGBA{hashFloat(1, x, y), hashFloat(2, x, y) * 3.f, -hashFloat(3, x, y), 1.f});
        OwnedImage ref = img, tmp(img.view.bounds);
        int radii[3];
        gaussianBoxRadii(sigma, radii);
        for (int r : radii) {
            if (r < 1) continue;
            refH(ref.view, tmp.view, r);
            refV(tmp.view, ref.view, r);
        }
        gaussianBlur(img, sigma);
        CHECK(maxDiff(img.view, ref.view) == 0.0);
    }
}

TP_TEST(plane_gaussian_blur_matches_rgba_blur) {
    for (float sigma : {0.9f, 6.3f, 30.f}) {
        const int w = 131, h = 47;
        OwnedImage img(Rect{0, 0, w, h});
        PixelBuffer plane(static_cast<size_t>(w) * h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                plane[static_cast<size_t>(y) * w + x] = hashFloat(9, x, y);
                img.view.store(x, y, RGBA{hashFloat(9, x, y), 0.f, 0.f, 0.f});
            }
        gaussianBlur(img, sigma);
        gaussianBlurPlane(plane.data(), w, h, sigma);
        double d = 0;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) d = std::max(d, static_cast<double>(std::fabs(img.view.load(x, y).r - plane[static_cast<size_t>(y) * w + x])));
        CHECK(d == 0.0);
    }
}
