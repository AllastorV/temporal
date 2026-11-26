#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace tp {

struct GlassPose {
    double x = 0, y = 0, depth = 0, rx = 0, ry = 0, rz = 0;
};

inline double glassNoiseHash(int64_t i, int seed) {
    uint32_t v = static_cast<uint32_t>(i) * 0x9e3779b1u ^ static_cast<uint32_t>(seed) * 0x85ebca6bu;
    v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
    return (v >> 8) * (2.0 / 16777216.0) - 1.0;
}

inline double glassNoise(double t, int seed) {
    const double f = std::floor(t), u = t - f, u2 = u * u, u3 = u2 * u;
    const int64_t i = static_cast<int64_t>(f);
    const double b0 = (1.0 - u) * (1.0 - u) * (1.0 - u) / 6.0, b1 = (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0;
    const double b2 = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0, b3 = u3 / 6.0;
    return 1.7 * (b0 * glassNoiseHash(i - 1, seed) + b1 * glassNoiseHash(i, seed) + b2 * glassNoiseHash(i + 1, seed) + b3 * glassNoiseHash(i + 2, seed));
}

inline double glassFbm(double t, int seed, int octaves, double rough) {
    double sum = 0, amp = 1, norm = 0, freq = 1;
    for (int o = 0; o < octaves; ++o) {
        sum += amp * glassNoise(t * freq + 7.31 * o, seed * 17 + o);
        norm += amp;
        amp *= rough;
        freq *= 2.13;
    }
    return sum / norm;
}

inline double glassSoftLimit(double v, double limit) {
    if (limit <= 0) return 0;
    const double knee = 0.55 * limit, a = std::fabs(v);
    if (a <= knee) return v;
    return std::copysign(knee + (limit - knee) * std::tanh((a - knee) / (limit - knee)), v);
}

inline void glassSoftLimit2(double& x, double& y, double limit) {
    const double r = std::hypot(x, y);
    if (r <= 0.55 * limit) return;
    const double k = glassSoftLimit(r, limit) / r;
    x *= k;
    y *= k;
}

inline GlassPose glassHandheld(int mode, double seconds, double rate, double amount, double turn) {
    GlassPose p;
    const double tau = 2.0 * 3.14159265358979, cycles = rate * seconds;
    const bool drift = mode == 4;
    const double scale = mode == 8 ? 0.5 : mode == 9 ? 1.8 : 1.0;
    const double tempo = mode == 8 ? 0.95 : mode == 9 ? 1.15 : 1.0;
    const double knots = drift ? 0.45 : 2.0 * tempo;
    const double shake = drift ? 0.0 : mode == 8 ? 0.006 : mode == 9 ? 0.012 : 0.015;
    const double tremor = drift ? 0.0 : (mode == 8 ? 0.5 : mode == 9 ? 1.6 : 0.8) * turn;
    const double breath = drift ? 0.0 : mode == 8 ? 0.2 : 0.14;
    const double roll = mode == 9 ? 2.0 : 1.0;
    const double depthScale = mode == 8 ? 0.5 : mode == 9 ? 1.3 : 1.0;
    const int octaves = 2, family = 100 * mode;
    const double t = knots * cycles, fine = 8.0 * seconds, lag = knots * rate * (drift ? 0.3 : 0.12);
    const auto sway = [&](double at, int seed) {
        const double now = glassFbm(at, seed + family, octaves, 0.35);
        return drift ? now : 0.8 * (now - glassFbm(at - 1.5, seed + family, octaves, 0.35));
    };
    const auto jitter = [&](double at, int seed) { return glassNoise(at, seed + family); };
    const double lungs = std::sin(tau * 0.25 * seconds * tempo + 0.4 * glassNoise(0.13 * seconds, 31)) * (0.7 + 0.3 * glassNoise(0.18 * seconds, 32));
    double x = sway(t, 1);
    double y = 0.45 * sway(t * 1.13 + 3.7, 2) + breath * lungs;
    if (mode == 9) {
        const double push = glassNoise(0.7 * seconds, 77 + family), heave = glassNoise(0.6 * seconds + 4.4, 78 + family);
        x += 0.25 * push * push * push;
        y += 0.15 * heave * heave * heave;
    }
    glassSoftLimit2(x, y, 1.0);
    p.x = amount * (scale * x + shake * jitter(fine * 1.21 + 3.3, 16));
    p.y = amount * (scale * y + shake * jitter(fine * 1.17 + 8.8, 17));
    p.depth = 0.25 * depthScale * glassSoftLimit(sway(t * 0.83 + 9.2, 3), 1.0);
    const double deg = (drift ? 4.0 : 6.0) * turn * scale;
    p.ry = deg * glassSoftLimit(0.45 * sway(t - lag, 1) + 0.55 * sway(t * 0.97 + 13.3, 5), 1.0) + tremor * jitter(fine * 0.9 + 2.2, 13);
    p.rx = -0.8 * deg * glassSoftLimit(0.35 * sway((t - lag) * 1.13 + 3.7, 2) + 0.65 * sway(t * 1.05 + 17.9, 4), 1.0) + tremor * jitter(fine * 1.1 + 7.7, 14);
    p.rz = 0.5 * roll * deg * glassSoftLimit(sway(t * 0.9 + 21.4, 6), 1.0) + 0.7 * roll * tremor * jitter(fine * 0.95 + 1.3, 15);
    return p;
}

inline GlassPose glassMotion(int mode, double seconds, double rate, double amount, double turn) {
    GlassPose p;
    if (mode <= 0 || amount <= 0) return p;
    if (mode == 1 || mode == 4 || mode == 8 || mode == 9) return glassHandheld(mode, seconds, rate, amount, turn);
    const double tau = 2.0 * 3.14159265358979, cycles = rate * seconds, lag = rate * 0.14;
    const double seed = 1.7 * mode;
    const auto phaseAt = [&](double c) {
        const double raw = tau * c + 0.2 * std::sin(tau * c / 3.3 + 0.3 + seed) + 0.12 * std::sin(tau * c / 5.9 + 0.7 + 2.3 * seed);
        return raw + 0.06 * std::sin(2.0 * raw);
    };
    const double phase = phaseAt(cycles), turned = phaseAt(cycles - lag);
    const double ax = 1.0 + 0.12 * std::sin(tau * cycles / 4.1 + 1.0 + seed) + 0.06 * std::sin(tau * cycles / 5.3 + 2.0 + 1.9 * seed);
    const double ay = 1.0 + 0.1 * std::sin(tau * cycles / 4.7 + 2.4 + 2.9 * seed) + 0.07 * std::sin(tau * cycles / 6.1 + 0.9 + 1.3 * seed);
    const double reach = amount * ax, deg = 8.0 * turn;
    if (mode == 2) {
        p.x = reach * std::sin(phase);
        p.y = 0.74 * amount * ay * std::cos(phase);
        p.depth = 0.3 * ax * std::sin(phase + 1.1);
        p.ry = ax * deg * std::cos(turned);
        p.rx = 0.74 * ay * deg * std::sin(turned);
        p.rz = 0.4 * deg * std::sin(turned);
    } else if (mode == 3) {
        p.x = reach * std::sin(phase);
        p.y = 0.24 * amount * ay * std::cos(phase);
        p.ry = 1.5 * ax * deg * std::cos(turned);
        p.rz = 0.4 * deg * std::sin(turned);
    } else if (mode == 5) {
        const double arc = 16.0 * tau / 360.0, length = amount / std::sin(arc);
        const double theta = arc * ax * std::sin(phase);
        p.x = length * std::sin(theta);
        p.y = length * (1.0 - std::cos(theta));
        p.rz = turn * theta * 360.0 / tau;
        p.ry = -0.5 * deg * std::sin(turned);
    } else if (mode == 6) {
        p.x = reach * std::sin(phase);
        p.y = 0.5 * amount * ay * std::sin(2.0 * phase);
        p.depth = 0.25 * ax * std::cos(phase);
        p.ry = ax * deg * std::cos(turned);
        p.rx = -0.6 * ay * deg * std::cos(2.0 * turned);
        p.rz = 0.5 * deg * std::sin(turned);
    } else if (mode == 7) {
        const double fold = 0.5 - 0.5 * std::cos(phase), folded = 0.5 - 0.5 * std::cos(turned);
        p.rx = 44.0 * turn * ax * folded;
        p.y = 0.25 * amount * ay * fold;
        p.ry = 0.5 * deg * std::sin(turned);
    }
    p.rx += 0.12 * deg * glassFbm(cycles * 1.3 + seed, 41, 2, 0.35);
    p.ry += 0.12 * deg * glassFbm(cycles * 1.3 + 5.5 + seed, 42, 2, 0.35);
    if (mode != 5) p.rz += 0.12 * deg * glassFbm(cycles * 1.3 + 9.1 + seed, 43, 2, 0.35);
    return p;
}

inline GlassPose glassMotionEased(int mode, double seconds, double rate, double amount, double turn, double ease) {
    GlassPose p = glassMotion(mode, seconds, rate, amount, turn);
    if (ease <= 0) return p;
    const double t = std::clamp(seconds / ease, 0.0, 1.0), k = t * t * (3.0 - 2.0 * t);
    p.x *= k; p.y *= k; p.depth *= k; p.rx *= k; p.ry *= k; p.rz *= k;
    return p;
}

}
