#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "rain_kernel.h"

namespace tp {

struct RainSettings {
    double seconds = 0;
    float aspect = 16.f / 9.f;
    float amount = 0.5f, minR = 0.001f, maxR = 0.01f, slideR = 0.007f;
    float slide = 0.35f, slideSpeed = 0.6f, wind = 0.f, wiggle = 0.5f;
    float trails = 0.6f, refog = 4.f, tempo = 1.f, shutter = 1.f / 60.f;
    uint32_t seed = 1;
};

inline void rainShape(RainDrop& d, uint32_t id, float rMax) {
    const auto rnd = [&](uint32_t k) { return rainHash(rainKey(id, 40 + k)); };
    const float big = rainSmooth(0.12f * rMax, rMax, d.r), pi = 3.14159265f;
    d.aspect = 1.f + (0.02f + 0.28f * big) * std::pow(rnd(0), 1.5f);
    const float orient = rnd(1) < 0.6f ? (rnd(2) - 0.5f) * 0.7f : rnd(2) * pi;
    d.orientCos = std::cos(orient);
    d.orientSin = std::sin(orient);
    d.sag = big * (0.08f + 0.14f * rnd(3));
    d.flat = big * (0.04f + 0.12f * rnd(4));
    const float sigma = 0.012f + 0.055f * big;
    static const auto falloff = [] {
        std::array<float, RainDrop::kHarmonics> f{};
        for (int j = 0; j < RainDrop::kHarmonics; ++j) f[j] = std::pow(static_cast<float>(kRainHarmonicOrder[j]) / 2.f, -1.6f);
        return f;
    }();
    for (int j = 0; j < RainDrop::kHarmonics; ++j) {
        const float k = static_cast<float>(kRainHarmonicOrder[j]);
        const float amp = (j < 6 ? sigma * falloff[j] : (0.003f + 0.007f * big) * 2.f / (k - 9.f)) * (0.3f + 1.4f * rnd(10 + j));
        const float phase = 2.f * pi * rnd(30 + j);
        d.ha[j] = amp * std::cos(phase);
        d.hb[j] = -amp * std::sin(phase);
    }
    d.cuts = big > 0.3f ? (rnd(5) < 0.45f ? 1 : rnd(5) < 0.75f ? 2 : 0) : 0;
    for (int j = 0; j < d.cuts; ++j) {
        const float a = rnd(50 + j) < 0.5f ? pi + (rnd(52 + j) - 0.5f) * 1.2f : 2.f * pi * rnd(52 + j);
        d.cutX[j] = std::cos(a);
        d.cutY[j] = std::sin(a);
        d.cutD[j] = 0.88f + 0.09f * rnd(54 + j);
    }
}

inline void rainAbsorbLobe(RainDrop& a, const RainDrop& b, float squeeze = 0.4f) {
    const float va = a.r * a.r * a.r, vb = b.r * b.r * b.r, r = std::cbrt(va + vb), wb = vb / std::max(1e-12f, va + vb);
    const float dx = b.x - a.x, dy = b.y - a.y;
    const float ax = -squeeze * dx * wb, ay = -squeeze * dy * wb, bx = squeeze * dx * (1.f - wb), by = squeeze * dy * (1.f - wb);
    const auto along = [&](float ox, float oy) { return ox * a.dirX + oy * a.dirY; };
    const auto across = [&](float ox, float oy) { return -ox * a.dirY + oy * a.dirX; };
    for (int i = 0; i < a.lobes; ++i) {
        a.lobeX[i] = (a.lobeX[i] * a.r + along(ax, ay)) / r;
        a.lobeY[i] = (a.lobeY[i] * a.r + across(ax, ay)) / r;
        a.lobeR[i] *= a.r / r;
    }
    if (a.lobes < RainDrop::kLobes) {
        a.lobeX[a.lobes] = along(bx, by) / r;
        a.lobeY[a.lobes] = across(bx, by) / r;
        a.lobeR[a.lobes] = b.r / r;
        ++a.lobes;
    }
    a.x += dx * wb;
    a.y += dy * wb;
    a.r = r;
    a.sag = std::max(a.sag, b.sag);
}

inline float rainSegmentDistance(float px, float py, const RainWipe& w) {
    const float ex = w.x1 - w.x0, ey = w.y1 - w.y0, len2 = std::max(1e-12f, ex * ex + ey * ey);
    const float t = std::clamp(((px - w.x0) * ex + (py - w.y0) * ey) / len2, 0.f, 1.f);
    const float dx = px - w.x0 - t * ex, dy = py - w.y0 - t * ey;
    return std::sqrt(dx * dx + dy * dy);
}

inline void rainSimulate(const RainSettings& s, std::vector<RainDrop>& drops, std::vector<RainWipe>& wipes) {
    drops.clear();
    wipes.clear();
    if (s.amount <= 0 || s.maxR <= 0) return;
    const float tempo = std::max(0.05f, s.tempo);
    const int slots = std::clamp(static_cast<int>(s.amount * 700.f * s.aspect), 0, 6000);
    const float rMax = s.maxR, rMin = std::clamp(s.minR, 0.02f * rMax, rMax), slideR = std::max(rMin, s.slideR);
    const float deg = 3.14159265f / 180.f;
    std::vector<int> runnerDrop;
    std::vector<float> runnerVolume;
    std::vector<RainDrop> ghosts;
    const auto run = [&](RainDrop& d, uint32_t id, double startAbs, float fade, int me, double death, float rPath) {
        const float age = static_cast<float>(s.seconds - startAbs) * tempo;
        if (age <= 0 || s.slideSpeed <= 0) return false;
        const float r0 = rPath, x0 = d.x, y0 = d.y;
        const float heft = std::sqrt(std::min(1.f, r0 / rMax));
        const float base = s.wind * (1.f - 0.35f * heft) + (rainHash(rainKey(id, 8)) - 0.5f) * 16.f;
        const float vmax = 0.45f * s.slideSpeed * (0.4f + 0.6f * heft);
        float px = x0, py = y0, clock = 0, speed = 0, travelled = 0;
        float dirX = std::sin(base * deg), dirY = -std::cos(base * deg);
        const float step = 0.8f * r0;
        float nextBead = step;
        for (int j = 0; j < 64; ++j) {
            const uint32_t seg = rainKey(id, 1000 + j);
            const float wait = j == 0 ? 0.f : (0.15f + 1.1f * rainHash(rainKey(seg, 1))) * (1.2f - 0.6f * heft);
            const float turn = base + (rainHash(rainKey(seg, 2)) - 0.5f) * 50.f * s.wiggle;
            const float dist = (0.025f + 0.11f * rainHash(rainKey(seg, 3))) * (0.6f + 0.6f * heft);
            const float dur = 1.6f * dist / std::max(1e-4f, vmax);
            const float sx = std::sin(turn * deg), sy = -std::cos(turn * deg);
            if (age < clock + wait) break;
            const float into = std::min(dur, age - clock - wait), u = into / dur;
            const float moved = dist * u * u * (3.f - 2.f * u);
            const float wasX = px, wasY = py;
            dirX = sx; dirY = sy;
            while (nextBead <= travelled + moved) {
                const float along = nextBead - travelled, bx = wasX + sx * along, by = wasY + sy * along;
                const double pass = startAbs + (clock + wait + dur * (along / std::max(1e-6f, dist))) / tempo;
                const float since = static_cast<float>(s.seconds - pass) * tempo;
                const uint32_t bead = rainKey(id, 5000 + static_cast<uint32_t>(nextBead / step));
                const float life = 1.5f + 4.f * rainHash(rainKey(bead, 1));
                const float keep = 1.f - rainSmooth(life - 0.6f, life, since);
                if (rainHash(rainKey(bead, 2)) < 0.75f * s.trails && keep > 0.f) {
                    RainDrop b;
                    const float side = 0.35f * r0 * (rainHash(rainKey(bead, 3)) - 0.5f);
                    b.x = bx - side * sy;
                    b.y = by + side * sx;
                    b.full = r0 * (0.1f + 0.18f * rainHash(rainKey(bead, 4)));
                    b.r = b.full * std::sqrt(keep * fade);
                    b.birth = pass;
                    b.death = std::min(death, pass + life / tempo);
                    b.bead = true;
                    rainShape(b, bead, rMax);
                    drops.push_back(b);
                }
                const float lane = 2.f + 3.f * rainHash(rainKey(bead, 6));
                const float clear = s.refog > 0 ? 1.f - rainSmooth(0.f, s.refog * lane / 3.5f, since / tempo) : 1.f;
                const float ragged = 0.8f + 0.4f * rainHash(rainKey(bead, 5));
                wipes.push_back({bx - sx * step, by - sy * step, bx, by, 0.75f * r0 * ragged, std::min(clear, fade), pass, death, me});
                nextBead += step;
            }
            px = wasX + sx * moved;
            py = wasY + sy * moved;
            travelled += moved;
            if (into < dur) {
                speed = 6.f * u * (1.f - u) * dist / dur;
                break;
            }
            clock += wait + dur;
            speed = 0;
        }
        d.x = px; d.y = py;
        d.r *= 1.f - 0.25f * rainSmooth(0.f, 0.6f, travelled);
        d.dirX = dirX; d.dirY = dirY;
        const float blur = speed * tempo * s.shutter;
        d.stretch = 1.f + 0.35f * rainSmooth(0.f, vmax, speed) + 0.5f * blur / std::max(1e-5f, d.r);
        d.tail = 0.8f * rainSmooth(0.f, vmax, speed);
        d.cuts = 0;
        d.runner = me;
        return true;
    };
    for (int k = 0; k < slots; ++k) {
        const uint32_t slot = rainKey(s.seed, static_cast<uint32_t>(k));
        const float period = (3.f + 5.f * rainHash(rainKey(slot, 1))) / tempo;
        const double t = s.seconds + rainHash(rainKey(slot, 2)) * period;
        const double cycle = std::floor(t / period);
        const float tau = static_cast<float>(t - cycle * period);
        const double born = s.seconds - tau;
        for (int back = 1; back >= 0; --back) {
            const uint32_t id = rainKey(slot, static_cast<uint32_t>(static_cast<int64_t>(cycle) - back) * 2654435761u);
            const double birth = born - back * period;
            const float h = rainHash(rainKey(id, 3));
            const float r0 = rMin * std::pow(rMax / rMin, std::pow(h, 1.7f));
            const bool runs = r0 > slideR || (r0 > 0.5f * slideR && rainHash(rainKey(id, 6)) < s.slide);
            RainDrop d;
            d.id = id;
            d.x = rainHash(rainKey(id, 4)) * s.aspect;
            d.y = rainHash(rainKey(id, 5)) * 1.1f;
            d.r = d.full = r0;
            d.birth = birth;
            d.death = birth + period;
            if (back == 1) {
                if (!runs) ghosts.push_back(d);
                continue;
            }
            const float grow = rainSmooth(0.f, 0.06f / tempo, tau);
            const float fade = 1.f - rainSmooth(period - 1.2f / tempo, period, tau);
            if (grow * fade <= 0.f) continue;
            rainShape(d, id, rMax);
            const float delay = r0 > slideR ? 0.15f + 0.5f * rainHash(rainKey(id, 7)) : period * (0.1f + 0.35f * rainHash(rainKey(id, 7)));
            d.willRun = runs;
            if (runs && run(d, id, birth + delay, fade, static_cast<int>(runnerDrop.size()), d.death, r0)) {
                runnerDrop.push_back(static_cast<int>(drops.size()));
                runnerVolume.push_back(d.r * d.r * d.r);
            }
            d.r *= std::sqrt(grow * fade);
            drops.push_back(d);
        }
    }
    std::vector<char> gone(drops.size(), 0);
    float lx = std::numeric_limits<float>::max(), ly = lx, hx = -lx, hy = -lx, maxFull = 0.f;
    for (const RainDrop& d : drops)
        if (d.runner == -1 && !d.willRun) {
            lx = std::min(lx, d.x); hx = std::max(hx, d.x);
            ly = std::min(ly, d.y); hy = std::max(hy, d.y);
            maxFull = std::max(maxFull, d.full);
        }
    if (lx <= hx && !wipes.empty()) {
        const float cs = std::max(2.f * rMax, std::max(hx - lx, hy - ly) / 255.f);
        const int gw = static_cast<int>((hx - lx) / cs) + 1, gh = static_cast<int>((hy - ly) / cs) + 1;
        const auto cellOf = [&](float v, float lo, int count) { return static_cast<int>(std::clamp(std::floor((v - lo) / cs), -1.f, static_cast<float>(count))); };
        const auto span = [&](const RainWipe& w, int& xa, int& xb, int& ya, int& yb) {
            const float reach = w.r + 0.4f * maxFull + 1e-5f;
            xa = std::max(0, cellOf(std::min(w.x0, w.x1) - reach, lx, gw));
            xb = std::min(gw - 1, cellOf(std::max(w.x0, w.x1) + reach, lx, gw));
            ya = std::max(0, cellOf(std::min(w.y0, w.y1) - reach, ly, gh));
            yb = std::min(gh - 1, cellOf(std::max(w.y0, w.y1) + reach, ly, gh));
            return w.runner >= 0 && xa <= xb && ya <= yb;
        };
        std::vector<int> cellStart(static_cast<size_t>(gw) * gh + 1, 0);
        for (const RainWipe& w : wipes) {
            int xa, xb, ya, yb;
            if (!span(w, xa, xb, ya, yb)) continue;
            for (int y = ya; y <= yb; ++y)
                for (int x = xa; x <= xb; ++x) ++cellStart[static_cast<size_t>(y) * gw + x + 1];
        }
        for (size_t c = 1; c < cellStart.size(); ++c) cellStart[c] += cellStart[c - 1];
        std::vector<int> cellWipes(static_cast<size_t>(cellStart.back())), fill(cellStart.begin(), cellStart.end() - 1);
        for (size_t k = 0; k < wipes.size(); ++k) {
            int xa, xb, ya, yb;
            if (!span(wipes[k], xa, xb, ya, yb)) continue;
            for (int y = ya; y <= yb; ++y)
                for (int x = xa; x <= xb; ++x) cellWipes[static_cast<size_t>(fill[static_cast<size_t>(y) * gw + x]++)] = static_cast<int>(k);
        }
        for (size_t i = 0; i < drops.size(); ++i) {
            const RainDrop& d = drops[i];
            if (d.runner != -1 || d.willRun) continue;
            const size_t c = static_cast<size_t>(cellOf(d.y, ly, gh)) * gw + static_cast<size_t>(cellOf(d.x, lx, gw));
            for (int q = cellStart[c]; q < cellStart[c + 1]; ++q) {
                const RainWipe& w = wipes[static_cast<size_t>(cellWipes[static_cast<size_t>(q)])];
                if (w.runner < 0 || w.pass <= d.birth || d.death > w.death) continue;
                if (rainSegmentDistance(d.x, d.y, w) < w.r + 0.4f * d.full) {
                    gone[i] = 1;
                    runnerVolume[static_cast<size_t>(w.runner)] += d.r * d.r * d.r;
                    break;
                }
            }
        }
    }
    for (size_t k = 0; k < runnerDrop.size(); ++k) {
        RainDrop& d = drops[static_cast<size_t>(runnerDrop[k])];
        d.r = std::min(1.5f * d.r, std::cbrt(runnerVolume[k]));
    }
    std::vector<RainDrop> nodes;
    std::vector<int> nodeDrop;
    nodes.reserve(ghosts.size() + drops.size());
    nodeDrop.reserve(ghosts.size() + drops.size());
    for (const RainDrop& g : ghosts) { nodes.push_back(g); nodeDrop.push_back(-1); }
    for (size_t i = 0; i < drops.size(); ++i)
        if (!gone[i] && drops[i].runner == -1 && !drops[i].bead && !drops[i].willRun) { nodes.push_back(drops[i]); nodeDrop.push_back(static_cast<int>(i)); }
    const size_t n = nodes.size();
    std::vector<int> parent(n), link(n, -1);
    std::vector<double> joined(n);
    for (size_t i = 0; i < n; ++i) { parent[i] = static_cast<int>(i); joined[i] = 1e30; }
    const auto root = [&](int i) { while (parent[static_cast<size_t>(i)] != i) i = parent[static_cast<size_t>(i)] = parent[static_cast<size_t>(parent[static_cast<size_t>(i)])]; return i; };
    const float cell = 2.f * rMax;
    std::vector<std::pair<int64_t, int>> grid;
    grid.reserve(n);
    const auto cellKey = [&](int cx, int cy) { return (static_cast<int64_t>(cx) << 32) ^ static_cast<int64_t>(static_cast<uint32_t>(cy)); };
    for (size_t i = 0; i < n; ++i)
        grid.push_back({cellKey(static_cast<int>(std::floor(nodes[i].x / cell)), static_cast<int>(std::floor(nodes[i].y / cell))), static_cast<int>(i)});
    std::sort(grid.begin(), grid.end());
    for (size_t i = 0; i < n; ++i) {
        const RainDrop& a = nodes[i];
        const int cx = static_cast<int>(std::floor(a.x / cell)), cy = static_cast<int>(std::floor(a.y / cell));
        for (int ox = -1; ox <= 1; ++ox)
            for (int oy = -1; oy <= 1; ++oy) {
                const bool column = cy >= 1;
                if (column && oy > -1) break;
                auto it = std::lower_bound(grid.begin(), grid.end(), std::make_pair(cellKey(cx + ox, cy + oy), std::numeric_limits<int>::min()));
                const int64_t last = cellKey(cx + ox, column ? cy + 1 : cy + oy);
                for (; it != grid.end() && it->first <= last; ++it) {
                    const size_t j = static_cast<size_t>(it->second);
                    if (j <= i) continue;
                    const RainDrop& b = nodes[j];
                    if (b.birth >= a.death || a.birth >= b.death) continue;
                    const float dx = b.x - a.x, dy = b.y - a.y, reach = 0.8f * (a.full + b.full);
                    if (dx * dx + dy * dy >= reach * reach) continue;
                    const double when = std::max(a.birth, b.birth);
                    joined[i] = std::min(joined[i], when);
                    joined[j] = std::min(joined[j], when);
                    parent[static_cast<size_t>(root(static_cast<int>(i)))] = root(static_cast<int>(j));
                }
            }
    }
    std::vector<std::vector<int>> groups(n);
    for (size_t i = 0; i < n; ++i) groups[static_cast<size_t>(root(static_cast<int>(i)))].push_back(static_cast<int>(i));
    for (auto& g : groups) {
        if (g.size() < 2) continue;
        std::sort(g.begin(), g.end(), [&](int a, int b) {
            const double ta = std::min(nodes[static_cast<size_t>(a)].birth, joined[static_cast<size_t>(a)]), tb = std::min(nodes[static_cast<size_t>(b)].birth, joined[static_cast<size_t>(b)]);
            return ta != tb ? ta < tb : nodes[static_cast<size_t>(a)].id < nodes[static_cast<size_t>(b)].id;
        });
        float volume = 0.f, sx = 0.f, sy = 0.f;
        double start = -1;
        size_t members = g.size();
        uint32_t runId = 0;
        for (size_t k = 0; k < g.size(); ++k) {
            const RainDrop& m = nodes[static_cast<size_t>(g[k])];
            const float v = m.full * m.full * m.full;
            volume += v; sx += m.x * v; sy += m.y * v;
            const double t = k == 0 ? m.birth : std::max(m.birth, joined[static_cast<size_t>(g[k])]);
            if (start < 0 && std::cbrt(volume) > slideR) {
                start = t;
                runId = rainKey(nodes[static_cast<size_t>(g[0])].id, m.id);
                members = k + 1;
                for (size_t q = k + 1; q < g.size(); ++q) {
                    const RainDrop& later = nodes[static_cast<size_t>(g[q])];
                    if (std::max(later.birth, joined[static_cast<size_t>(g[q])]) <= start + 0.08 / tempo) members = q + 1;
                }
                break;
            }
        }
        int host = -1;
        for (size_t k = 0; k < members; ++k) {
            const int di = nodeDrop[static_cast<size_t>(g[k])];
            if (di >= 0 && (host < 0 || drops[static_cast<size_t>(di)].death > drops[static_cast<size_t>(host)].death)) host = di;
        }
        if (host < 0) continue;
        RainDrop merged = drops[static_cast<size_t>(host)];
        for (size_t k = 0; k < members; ++k) {
            const int di = nodeDrop[static_cast<size_t>(g[k])];
            if (di < 0 || di == host) continue;
            const float since = static_cast<float>(s.seconds - std::max(drops[static_cast<size_t>(di)].birth, joined[static_cast<size_t>(g[k])])) * tempo;
            rainAbsorbLobe(merged, drops[static_cast<size_t>(di)], 1.f - 0.6f * rainSmooth(0.f, 0.3f, since));
            gone[static_cast<size_t>(di)] = 1;
        }
        if (start >= 0) {
            float cx = 0.f, cy = 0.f, cv = 0.f;
            for (size_t k = 0; k < members; ++k) {
                const RainDrop& m = nodes[static_cast<size_t>(g[k])];
                const float v = m.full * m.full * m.full;
                cx += m.x * v; cy += m.y * v; cv += v;
            }
            const float shiftX = cx / cv - merged.x, shiftY = cy / cv - merged.y;
            RainDrop d = merged;
            d.x += shiftX;
            d.y += shiftY;
            for (int l = 0; l < d.lobes; ++l) {
                d.lobeX[l] -= (shiftX * d.dirX + shiftY * d.dirY) / d.r;
                d.lobeY[l] -= (-shiftX * d.dirY + shiftY * d.dirX) / d.r;
            }
            const float fade = rainSmooth(0.f, 1.2f, static_cast<float>(merged.death - s.seconds) * tempo);
            if (run(d, runId, start + 0.08 / tempo, fade, -2, merged.death, std::cbrt(cv))) {
                merged = d;
                gone.resize(drops.size(), 0);
            }
        }
        drops[static_cast<size_t>(host)] = merged;
    }
    size_t keep = 0;
    for (size_t i = 0; i < drops.size(); ++i)
        if (i >= gone.size() || !gone[i]) drops[keep++] = drops[i];
    drops.resize(keep);
}

}
