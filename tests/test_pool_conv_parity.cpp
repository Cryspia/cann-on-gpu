// Cross-check for the ~23 pooling/vision/conv "gap" ops added in metal/src/ops/pool_conv_parity.mm.
// CPU double-precision reference per op, small shapes. Tolerance 1e-5 (exact for integer indices).
//
// Conventions documented inline:
//   nearest:        ih = floor(o*isz/osz)          (V2 / 2d / 3d nearest)
//   nearest-exact:  ih = floor((o+0.5)*isz/osz)    (half-pixel sampling), all clamped to [0,isz-1]
//   linear/bilinear/bicubic align_corners:
//       align=1 -> src = o*(isz-1)/(osz-1)         (endpoints map exactly)
//       align=0 -> src = (o+0.5)*isz/osz - 0.5     (half-pixel; clamped >=0)
//   grid_sample align_corners:
//       align=1 -> fx = (gx+1)/2*(W-1)
//       align=0 -> fx = ((gx+1)*W - 1)/2 ; bilinear/trilinear, zeros padding outside.
//   *AA variants intentionally fall back to non-AA (documented backend limitation; matches CUDA ref).
//   MultiScaleDeformableAttn samplingLocations are normalized in [0,1]: x=loc_x*W-0.5, y=loc_y*H-0.5.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <algorithm>
#include <cmath>
#include <vector>
using namespace hn;

static aclIntArray *ia(std::vector<int64_t> v) { return aclCreateIntArray(v.data(), v.size()); }

static int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
static int nsrc(int o, int isz, int osz, bool exact) {
    int i = exact ? (int)std::floor((o + 0.5f) * isz / osz) : (int)std::floor((float)o * isz / osz);
    return clampi(i, 0, isz - 1);
}
static void lcoord(int o, int isz, int osz, bool align, int &lo, int &hi, float &fr) {
    float s; if (align) s = osz > 1 ? (float)o * (isz - 1) / (osz - 1) : 0.f;
    else s = osz > 0 ? ((o + 0.5f) * isz / osz - 0.5f) : 0.f;
    if (s < 0) s = 0; int b = (int)std::floor(s); fr = s - b; lo = clampi(b, 0, isz - 1); hi = clampi(b + 1, 0, isz - 1);
}
static float cubicw(float t) { t = std::fabs(t); float a = -0.75f;
    if (t <= 1) return ((a + 2) * t - (a + 3)) * t * t + 1; if (t < 2) return (((t - 5) * t + 8) * t - 4) * a; return 0.f; }
static float bilin(const float *p, int H, int W, float fy, float fx) {
    if (fy < -1 || fy > H || fx < -1 || fx > W) return 0.f; fy = std::max(fy, 0.f); fx = std::max(fx, 0.f);
    int y0 = (int)fy, x0 = (int)fx, y1 = std::min(y0 + 1, H - 1), x1 = std::min(x0 + 1, W - 1); float dy = fy - y0, dx = fx - x0;
    return (1 - dy) * (1 - dx) * p[y0 * W + x0] + (1 - dy) * dx * p[y0 * W + x1]
         + dy * (1 - dx) * p[y1 * W + x0] + dy * dx * p[y1 * W + x1]; }

// ---- nearest (1d/2d/3d) generic, interp = floor(o*isz/osz) for V2/non-exact ----
static void cpu_nearest(const std::vector<float> &in, std::vector<double> &ref, int NC, int i0, int i1, int i2,
                        int o0, int o1, int o2, bool exact) {
    int64_t isp = (int64_t)i0 * i1 * i2, osp = (int64_t)o0 * o1 * o2; ref.assign((size_t)NC * osp, 0);
    for (int nc = 0; nc < NC; nc++) for (int c0 = 0; c0 < o0; c0++) for (int c1 = 0; c1 < o1; c1++) for (int c2 = 0; c2 < o2; c2++) {
        int s0 = nsrc(c0, i0, o0, exact), s1 = nsrc(c1, i1, o1, exact), s2 = nsrc(c2, i2, o2, exact);
        ref[nc * osp + ((int64_t)c0 * o1 + c1) * o2 + c2] = in[nc * isp + ((int64_t)s0 * i1 + s1) * i2 + s2]; }
}
static void cpu_linear(const std::vector<float> &in, std::vector<double> &ref, int NC, int i0, int i1, int i2,
                       int o0, int o1, int o2, bool align) {
    int64_t isp = (int64_t)i0 * i1 * i2, osp = (int64_t)o0 * o1 * o2; ref.assign((size_t)NC * osp, 0);
    for (int nc = 0; nc < NC; nc++) for (int c0 = 0; c0 < o0; c0++) for (int c1 = 0; c1 < o1; c1++) for (int c2 = 0; c2 < o2; c2++) {
        int l0, h0, l1, h1, l2, h2; float f0, f1, f2;
        lcoord(c0, i0, o0, align, l0, h0, f0); lcoord(c1, i1, o1, align, l1, h1, f1); lcoord(c2, i2, o2, align, l2, h2, f2);
        double acc = 0; const float *p = &in[nc * isp];
        for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int c = 0; c < 2; c++) {
            int z = a ? h0 : l0, y = b ? h1 : l1, x = c ? h2 : l2; double w = (a ? f0 : 1 - f0) * (b ? f1 : 1 - f1) * (c ? f2 : 1 - f2);
            acc += w * p[((int64_t)z * i1 + y) * i2 + x]; }
        ref[nc * osp + ((int64_t)c0 * o1 + c1) * o2 + c2] = acc; }
}

// =================================================================================================
static void t_upsample_nearest() {
    // 1d: [1,2,4]->[1,2,7]; 3d: [1,1,2,2,2]->[1,1,4,3,5]; 2dV2 nearest; nearest-exact 1/2/3d
    { int C = 2, L = 4, oL = 7; auto v = randv(C * L, -2, 2); std::vector<float> hz(C * oL);
      DevBuf da(C * L * 4), dz(C * oL * 4); da.up(v.data());
      aclTensor *ta = mk({1, C, L}, ACL_FLOAT, da.p), *tz = mk({1, C, oL}, ACL_FLOAT, dz.p);
      exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnUpsampleNearest1dV2GetWorkspaceSize(ta, tz, w, e); }, aclnnUpsampleNearest1dV2);
      dz.down(hz.data()); std::vector<double> ref; cpu_nearest(v, ref, C, 1, 1, L, 1, 1, oL, false);
      report("UpsampleNearest1dV2", norm_err(hz, ref), 1e-5); aclDestroyTensor(ta); aclDestroyTensor(tz); }

    { int C = 2, H = 3, W = 3, oH = 5, oW = 6; auto v = randv(C * H * W, -2, 2); std::vector<float> hz(C * oH * oW);
      DevBuf da(C * H * W * 4), dz(C * oH * oW * 4); da.up(v.data());
      aclTensor *ta = mk({1, C, H, W}, ACL_FLOAT, da.p), *tz = mk({1, C, oH, oW}, ACL_FLOAT, dz.p);
      exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnUpsampleNearest2dV2GetWorkspaceSize(ta, tz, w, e); }, aclnnUpsampleNearest2dV2);
      dz.down(hz.data()); std::vector<double> ref(C * oH * oW);
      for (int c = 0; c < C; c++) for (int oh = 0; oh < oH; oh++) for (int ow = 0; ow < oW; ow++) {
          int ih = std::min(oh * H / oH, H - 1), iw = std::min(ow * W / oW, W - 1); ref[(c * oH + oh) * oW + ow] = v[(c * H + ih) * W + iw]; }
      report("UpsampleNearest2dV2", norm_err(hz, ref), 1e-5); aclDestroyTensor(ta); aclDestroyTensor(tz); }

    { int C = 1, D = 2, H = 2, W = 2, oD = 4, oH = 3, oW = 5; auto v = randv(C * D * H * W, -2, 2); std::vector<float> hz(C * oD * oH * oW);
      DevBuf da(C * D * H * W * 4), dz(C * oD * oH * oW * 4); da.up(v.data());
      aclTensor *ta = mk({1, C, D, H, W}, ACL_FLOAT, da.p), *tz = mk({1, C, oD, oH, oW}, ACL_FLOAT, dz.p);
      exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnUpsampleNearest3dGetWorkspaceSize(ta, tz, w, e); }, aclnnUpsampleNearest3d);
      dz.down(hz.data()); std::vector<double> ref; cpu_nearest(v, ref, C, D, H, W, oD, oH, oW, false);
      report("UpsampleNearest3d", norm_err(hz, ref), 1e-5); aclDestroyTensor(ta); aclDestroyTensor(tz); }

    // nearest-exact 1/2/3d (half-pixel)
    { int C = 2, L = 4, oL = 7; auto v = randv(C * L, -2, 2); std::vector<float> hz(C * oL);
      DevBuf da(C * L * 4), dz(C * oL * 4); da.up(v.data());
      aclTensor *ta = mk({1, C, L}, ACL_FLOAT, da.p), *tz = mk({1, C, oL}, ACL_FLOAT, dz.p);
      exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnUpsampleNearestExact1dGetWorkspaceSize(ta, tz, w, e); }, aclnnUpsampleNearestExact1d);
      dz.down(hz.data()); std::vector<double> ref; cpu_nearest(v, ref, C, 1, 1, L, 1, 1, oL, true);
      report("UpsampleNearestExact1d", norm_err(hz, ref), 1e-5); aclDestroyTensor(ta); aclDestroyTensor(tz); }
    { int C = 2, H = 3, W = 4, oH = 5, oW = 7; auto v = randv(C * H * W, -2, 2); std::vector<float> hz(C * oH * oW);
      DevBuf da(C * H * W * 4), dz(C * oH * oW * 4); da.up(v.data());
      aclTensor *ta = mk({1, C, H, W}, ACL_FLOAT, da.p), *tz = mk({1, C, oH, oW}, ACL_FLOAT, dz.p);
      exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnUpsampleNearestExact2dGetWorkspaceSize(ta, tz, w, e); }, aclnnUpsampleNearestExact2d);
      dz.down(hz.data()); std::vector<double> ref; cpu_nearest(v, ref, C, 1, H, W, 1, oH, oW, true);
      report("UpsampleNearestExact2d", norm_err(hz, ref), 1e-5); aclDestroyTensor(ta); aclDestroyTensor(tz); }
    { int C = 1, D = 2, H = 2, W = 3, oD = 3, oH = 4, oW = 5; auto v = randv(C * D * H * W, -2, 2); std::vector<float> hz(C * oD * oH * oW);
      DevBuf da(C * D * H * W * 4), dz(C * oD * oH * oW * 4); da.up(v.data());
      aclTensor *ta = mk({1, C, D, H, W}, ACL_FLOAT, da.p), *tz = mk({1, C, oD, oH, oW}, ACL_FLOAT, dz.p);
      exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnUpsampleNearestExact3dGetWorkspaceSize(ta, tz, w, e); }, aclnnUpsampleNearestExact3d);
      dz.down(hz.data()); std::vector<double> ref; cpu_nearest(v, ref, C, D, H, W, oD, oH, oW, true);
      report("UpsampleNearestExact3d", norm_err(hz, ref), 1e-5); aclDestroyTensor(ta); aclDestroyTensor(tz); }
}

static void t_upsample_linear() {
    for (int align = 0; align <= 1; align++) {
        int C = 2, L = 4, oL = 9; auto v = randv(C * L, -2, 2); std::vector<float> hz(C * oL);
        DevBuf da(C * L * 4), dz(C * oL * 4); da.up(v.data());
        aclTensor *ta = mk({1, C, L}, ACL_FLOAT, da.p), *tz = mk({1, C, oL}, ACL_FLOAT, dz.p);
        exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnUpsampleLinear1dGetWorkspaceSize(ta, (bool)align, tz, w, e); }, aclnnUpsampleLinear1d);
        dz.down(hz.data()); std::vector<double> ref; cpu_linear(v, ref, C, 1, 1, L, 1, 1, oL, align);
        report(std::string("UpsampleLinear1d align=") + std::to_string(align), norm_err(hz, ref), 1e-5);
        aclDestroyTensor(ta); aclDestroyTensor(tz);
    }
    for (int align = 0; align <= 1; align++) {
        int C = 1, D = 2, H = 3, W = 2, oD = 4, oH = 5, oW = 4; auto v = randv(C * D * H * W, -2, 2); std::vector<float> hz(C * oD * oH * oW);
        DevBuf da(C * D * H * W * 4), dz(C * oD * oH * oW * 4); da.up(v.data());
        aclTensor *ta = mk({1, C, D, H, W}, ACL_FLOAT, da.p), *tz = mk({1, C, oD, oH, oW}, ACL_FLOAT, dz.p);
        exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnUpsampleTrilinear3dGetWorkspaceSize(ta, (bool)align, tz, w, e); }, aclnnUpsampleTrilinear3d);
        dz.down(hz.data()); std::vector<double> ref; cpu_linear(v, ref, C, D, H, W, oD, oH, oW, align);
        report(std::string("UpsampleTrilinear3d align=") + std::to_string(align), norm_err(hz, ref), 1e-5);
        aclDestroyTensor(ta); aclDestroyTensor(tz);
    }
}

static void t_upsample_2d() {
    int C = 2, H = 3, W = 3, oH = 6, oW = 5; auto v = randv(C * H * W, -2, 2);
    for (int align = 0; align <= 1; align++) {
        std::vector<float> hz(C * oH * oW); DevBuf da(C * H * W * 4), dz(C * oH * oW * 4); da.up(v.data());
        aclTensor *ta = mk({1, C, H, W}, ACL_FLOAT, da.p), *tz = mk({1, C, oH, oW}, ACL_FLOAT, dz.p);
        // Bilinear2D and Bilinear2dAA share the same bilinear math; check both against the same ref.
        exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnUpsampleBilinear2DGetWorkspaceSize(ta, (bool)align, tz, w, e); }, aclnnUpsampleBilinear2D);
        dz.down(hz.data()); std::vector<double> ref; cpu_linear(v, ref, C, 1, H, W, 1, oH, oW, align);
        report(std::string("UpsampleBilinear2D align=") + std::to_string(align), norm_err(hz, ref), 1e-5);

        std::vector<float> hz2(C * oH * oW);
        exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnUpsampleBilinear2dAAGetWorkspaceSize(ta, (bool)align, tz, w, e); }, aclnnUpsampleBilinear2dAA);
        dz.down(hz2.data());
        report(std::string("UpsampleBilinear2dAA align=") + std::to_string(align), norm_err(hz2, ref), 1e-5);
        aclDestroyTensor(ta); aclDestroyTensor(tz);
    }
    // Bicubic2dAA
    for (int align = 0; align <= 1; align++) {
        std::vector<float> hz(C * oH * oW); DevBuf da(C * H * W * 4), dz(C * oH * oW * 4); da.up(v.data());
        aclTensor *ta = mk({1, C, H, W}, ACL_FLOAT, da.p), *tz = mk({1, C, oH, oW}, ACL_FLOAT, dz.p);
        exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnUpsampleBicubic2dAAGetWorkspaceSize(ta, (bool)align, tz, w, e); }, aclnnUpsampleBicubic2dAA);
        dz.down(hz.data()); std::vector<double> ref(C * oH * oW);
        for (int c = 0; c < C; c++) for (int oh = 0; oh < oH; oh++) for (int ow = 0; ow < oW; ow++) {
            float fh, fw; if (align) { fh = oH > 1 ? (float)oh * (H - 1) / (oH - 1) : 0; fw = oW > 1 ? (float)ow * (W - 1) / (oW - 1) : 0; }
            else { fh = (oh + 0.5f) * H / oH - 0.5f; fw = (ow + 0.5f) * W / oW - 0.5f; }
            int y0 = (int)std::floor(fh), x0 = (int)std::floor(fw); float dy = fh - y0, dx = fw - x0; const float *p = &v[c * H * W]; double acc = 0;
            for (int m = -1; m <= 2; m++) { float wy = cubicw(dy - m); int yy = clampi(y0 + m, 0, H - 1);
                for (int n = -1; n <= 2; n++) { float wx = cubicw(dx - n); int xx = clampi(x0 + n, 0, W - 1); acc += (double)wy * wx * p[yy * W + xx]; } }
            ref[(c * oH + oh) * oW + ow] = acc; }
        report(std::string("UpsampleBicubic2dAA align=") + std::to_string(align), norm_err(hz, ref), 1e-5);
        aclDestroyTensor(ta); aclDestroyTensor(tz);
    }
}

static void t_upsample_bwd() {
    // UpsampleBilinear2dBackwardV2: scatter bilinear weights of gradOut into gradIn.
    for (int align = 0; align <= 1; align++) {
        int C = 2, H = 3, W = 3, oH = 5, oW = 4; auto go = randv(C * oH * oW, -2, 2); std::vector<float> hz(C * H * W);
        DevBuf dgo(C * oH * oW * 4), dgi(C * H * W * 4); dgo.up(go.data());
        aclTensor *tgo = mk({1, C, oH, oW}, ACL_FLOAT, dgo.p), *tgi = mk({1, C, H, W}, ACL_FLOAT, dgi.p);
        exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnUpsampleBilinear2dBackwardV2GetWorkspaceSize(tgo, (bool)align, tgi, w, e); }, aclnnUpsampleBilinear2dBackwardV2);
        dgi.down(hz.data()); std::vector<double> ref(C * H * W, 0);
        for (int c = 0; c < C; c++) for (int oh = 0; oh < oH; oh++) for (int ow = 0; ow < oW; ow++) {
            float fh, fw; if (align) { fh = oH > 1 ? (float)oh * (H - 1) / (oH - 1) : 0; fw = oW > 1 ? (float)ow * (W - 1) / (oW - 1) : 0; }
            else { fh = (oh + 0.5f) * H / oH - 0.5f; fw = (ow + 0.5f) * W / oW - 0.5f; }
            fh = fh < 0 ? 0 : fh; fw = fw < 0 ? 0 : fw; int h0 = (int)fh, w0 = (int)fw, h1 = std::min(h0 + 1, H - 1), w1 = std::min(w0 + 1, W - 1);
            float dh = fh - h0, dw = fw - w0, g = go[(c * oH + oh) * oW + ow]; double *q = &ref[c * H * W];
            q[h0 * W + w0] += (1 - dh) * (1 - dw) * g; q[h0 * W + w1] += (1 - dh) * dw * g; q[h1 * W + w0] += dh * (1 - dw) * g; q[h1 * W + w1] += dh * dw * g; }
        report(std::string("UpsampleBilinear2dBackwardV2 align=") + std::to_string(align), norm_err(hz, ref), 1e-5);
        aclDestroyTensor(tgo); aclDestroyTensor(tgi);
    }
}

static void t_pool() {
    // GlobalMaxPool: [1,3,4,5] -> [1,3,1,1]
    { int C = 3, H = 4, W = 5; auto v = randv(C * H * W, -2, 2); std::vector<float> hz(C);
      DevBuf da(C * H * W * 4), dz(C * 4); da.up(v.data());
      aclTensor *ta = mk({1, C, H, W}, ACL_FLOAT, da.p), *tz = mk({1, C, 1, 1}, ACL_FLOAT, dz.p);
      exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnGlobalMaxPoolGetWorkspaceSize(ta, tz, w, e); }, aclnnGlobalMaxPool);
      dz.down(hz.data()); std::vector<double> ref(C);
      for (int c = 0; c < C; c++) { double best = -1e30; for (int i = 0; i < H * W; i++) best = std::max(best, (double)v[c * H * W + i]); ref[c] = best; }
      report("GlobalMaxPool", norm_err(hz, ref), 1e-5); aclDestroyTensor(ta); aclDestroyTensor(tz); }

    // MaxPool generic: [1,2,6,6], k=3 s=2 p=1 -> oH=oW=3
    { int C = 2, H = 6, W = 6, k = 3, st = 2, pad = 1; int oH = (H + 2 * pad - k) / st + 1, oW = (W + 2 * pad - k) / st + 1;
      auto v = randv(C * H * W, -2, 2); std::vector<float> hz(C * oH * oW);
      DevBuf da(C * H * W * 4), dz(C * oH * oW * 4); da.up(v.data());
      aclTensor *ta = mk({1, C, H, W}, ACL_FLOAT, da.p), *tz = mk({1, C, oH, oW}, ACL_FLOAT, dz.p);
      aclIntArray *ker = ia({k, k}), *str = ia({st, st}), *pd = ia({pad, pad}), *dl = ia({1, 1});
      exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnMaxPoolGetWorkspaceSize(ta, ker, str, pd, dl, false, tz, w, e); }, aclnnMaxPool);
      dz.down(hz.data()); std::vector<double> ref(C * oH * oW);
      for (int c = 0; c < C; c++) for (int oh = 0; oh < oH; oh++) for (int ow = 0; ow < oW; ow++) {
          double best = -1e30; for (int a = 0; a < k; a++) for (int b = 0; b < k; b++) { int ih = oh * st - pad + a, iw = ow * st - pad + b;
              if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue; best = std::max(best, (double)v[(c * H + ih) * W + iw]); } ref[(c * oH + oh) * oW + ow] = best; }
      report("MaxPool", norm_err(hz, ref), 1e-5); aclDestroyTensor(ta); aclDestroyTensor(tz);
      aclDestroyIntArray(ker); aclDestroyIntArray(str); aclDestroyIntArray(pd); aclDestroyIntArray(dl); }

    // MaxPool2dWithMask: value + flat index (ih*W+iw). Use distinct values to make argmax unambiguous.
    { int C = 1, H = 4, W = 4, k = 2, st = 2, pad = 0; int oH = 2, oW = 2;
      std::vector<float> v(C * H * W); for (int i = 0; i < C * H * W; i++) v[i] = (float)(i) * 0.5f + 0.1f;
      std::vector<float> hz(C * oH * oW); std::vector<int64_t> hi(C * oH * oW);
      DevBuf da(C * H * W * 4), dz(C * oH * oW * 4), di(C * oH * oW * 8); da.up(v.data());
      aclTensor *ta = mk({1, C, H, W}, ACL_FLOAT, da.p), *tz = mk({1, C, oH, oW}, ACL_FLOAT, dz.p), *tm = mk({1, C, oH, oW}, ACL_INT64, di.p);
      aclIntArray *ker = ia({k, k}), *str = ia({st, st}), *pd = ia({pad, pad}), *dl = ia({1, 1});
      exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnMaxPool2dWithMaskGetWorkspaceSize(ta, ker, str, pd, dl, false, tz, tm, w, e); }, aclnnMaxPool2dWithMask);
      dz.down(hz.data()); di.down(hi.data()); double vbad = 0; int ibad = 0;
      for (int c = 0; c < C; c++) for (int oh = 0; oh < oH; oh++) for (int ow = 0; ow < oW; ow++) {
          double best = -1e30; int64_t bi = 0; for (int a = 0; a < k; a++) for (int b = 0; b < k; b++) { int ih = oh * st - pad + a, iw = ow * st - pad + b;
              if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue; double val = v[(c * H + ih) * W + iw]; if (val > best) { best = val; bi = ih * W + iw; } }
          int oi = (c * oH + oh) * oW + ow; vbad = std::max(vbad, std::fabs(hz[oi] - best)); if (hi[oi] != bi) ibad++; }
      report("MaxPool2dWithMask val", vbad, 1e-5); report("MaxPool2dWithMask idx", (double)ibad, 0);
      aclDestroyTensor(ta); aclDestroyTensor(tz); aclDestroyTensor(tm);
      aclDestroyIntArray(ker); aclDestroyIntArray(str); aclDestroyIntArray(pd); aclDestroyIntArray(dl); }
}

static void t_roi() {
    // RoiAlignV2: in[1,2,8,8], 2 rois [batch,x1,y1,x2,y2], scale=1, ratio=2 -> out[2,2,2,2]
    int C = 2, H = 8, W = 8, K = 2, ph = 2, pw = 2, ratio = 2; float scale = 1.f;
    auto v = randv(C * H * W, -2, 2); float rois[10] = {0, 1, 1, 5, 5, 0, 2, 0, 7, 6};
    std::vector<float> hz(K * C * ph * pw);
    DevBuf da(C * H * W * 4), dr(K * 5 * 4), dz(K * C * ph * pw * 4); da.up(v.data()); dr.up(rois);
    aclTensor *ta = mk({1, C, H, W}, ACL_FLOAT, da.p), *tr = mk({K, 5}, ACL_FLOAT, dr.p), *tz = mk({K, C, ph, pw}, ACL_FLOAT, dz.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnRoiAlignV2GetWorkspaceSize(ta, tr, scale, ratio, tz, w, e); }, aclnnRoiAlignV2);
    dz.down(hz.data()); std::vector<double> ref(K * C * ph * pw);
    for (int k = 0; k < K; k++) for (int c = 0; c < C; c++) for (int py = 0; py < ph; py++) for (int px = 0; px < pw; px++) {
        const float *r = rois + k * 5; int b = (int)r[0]; float x1 = r[1] * scale, y1 = r[2] * scale, x2 = r[3] * scale, y2 = r[4] * scale;
        float rw = std::max(x2 - x1, 1.f), rh = std::max(y2 - y1, 1.f), bw = rw / pw, bh = rh / ph; const float *p = &v[(b * C + c) * H * W];
        double sum = 0; int cnt = ratio * ratio; for (int iy = 0; iy < ratio; iy++) for (int ix = 0; ix < ratio; ix++) {
            float yy = y1 + py * bh + (iy + 0.5f) * bh / ratio, xx = x1 + px * bw + (ix + 0.5f) * bw / ratio; sum += bilin(p, H, W, yy, xx); }
        ref[((k * C + c) * ph + py) * pw + px] = sum / cnt; }
    report("RoiAlignV2", norm_err(hz, ref), 1e-5); aclDestroyTensor(ta); aclDestroyTensor(tr); aclDestroyTensor(tz);

    // RoiPoolingGradWithArgMax: gradOut[K,C,ph,pw] + argmax (flat in H*W) -> gradInput[N,C,H,W] scatter-add.
    int N2 = 1, C2 = 2, H2 = 6, W2 = 6, K2 = 2, ph2 = 2, pw2 = 2;
    auto go = randv(K2 * C2 * ph2 * pw2, -2, 2); float rois2[10] = {0, 0, 0, 4, 4, 0, 1, 1, 5, 5};
    std::vector<int64_t> am(K2 * C2 * ph2 * pw2); for (size_t i = 0; i < am.size(); i++) am[i] = (int64_t)((i * 7 + 3) % (H2 * W2));
    std::vector<float> hz2(N2 * C2 * H2 * W2);
    DevBuf dgo(K2 * C2 * ph2 * pw2 * 4), dr2(K2 * 5 * 4), dam(am.size() * 8), dgi(N2 * C2 * H2 * W2 * 4);
    dgo.up(go.data()); dr2.up(rois2); dam.up(am.data());
    aclTensor *tgo = mk({K2, C2, ph2, pw2}, ACL_FLOAT, dgo.p), *tr2 = mk({K2, 5}, ACL_FLOAT, dr2.p),
              *tam = mk({K2, C2, ph2, pw2}, ACL_INT64, dam.p), *tgi = mk({N2, C2, H2, W2}, ACL_FLOAT, dgi.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize(tgo, tr2, tam, tgi, w, e); }, aclnnRoiPoolingGradWithArgMax);
    dgi.down(hz2.data()); std::vector<double> ref2(N2 * C2 * H2 * W2, 0);
    for (int k = 0; k < K2; k++) { const float *r = rois2 + k * 5; int b = (int)r[0];
        for (int c = 0; c < C2; c++) for (int py = 0; py < ph2; py++) for (int px = 0; px < pw2; px++) {
            int oi = ((k * C2 + c) * ph2 + py) * pw2 + px; int64_t fi = am[oi]; ref2[(b * C2 + c) * H2 * W2 + fi] += go[oi]; } }
    report("RoiPoolingGradWithArgMax", norm_err(hz2, ref2), 1e-5);
    aclDestroyTensor(tgo); aclDestroyTensor(tr2); aclDestroyTensor(tam); aclDestroyTensor(tgi);
}

static void t_gridsampler() {
    for (int align = 0; align <= 1; align++) {
        int N = 1, C = 2, H = 4, W = 4, oH = 3, oW = 3; auto v = randv(N * C * H * W, -2, 2); auto grid = randv(N * oH * oW * 2, -1, 1);
        std::vector<float> hz(N * C * oH * oW); DevBuf da(N * C * H * W * 4), dg(N * oH * oW * 2 * 4), dz(N * C * oH * oW * 4); da.up(v.data()); dg.up(grid.data());
        aclTensor *ta = mk({N, C, H, W}, ACL_FLOAT, da.p), *tg = mk({N, oH, oW, 2}, ACL_FLOAT, dg.p), *tz = mk({N, C, oH, oW}, ACL_FLOAT, dz.p);
        exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnGridSampler2DGetWorkspaceSize(ta, tg, 0, 0, (bool)align, tz, w, e); }, aclnnGridSampler2D);
        dz.down(hz.data()); std::vector<double> ref(N * C * oH * oW);
        for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) for (int oh = 0; oh < oH; oh++) for (int ow = 0; ow < oW; ow++) {
            const float *gg = &grid[((n * oH + oh) * oW + ow) * 2]; float gx = gg[0], gy = gg[1], fx, fy;
            if (align) { fx = (gx + 1) * 0.5f * (W - 1); fy = (gy + 1) * 0.5f * (H - 1); } else { fx = ((gx + 1) * W - 1) * 0.5f; fy = ((gy + 1) * H - 1) * 0.5f; }
            int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy), x1 = x0 + 1, y1 = y0 + 1; float dx = fx - x0, dy = fy - y0; const float *p = &v[(n * C + c) * H * W];
            auto at = [&](int y, int x) -> double { return (y >= 0 && y < H && x >= 0 && x < W) ? p[y * W + x] : 0.0; };
            ref[((n * C + c) * oH + oh) * oW + ow] = at(y0, x0) * (1 - dy) * (1 - dx) + at(y0, x1) * (1 - dy) * dx + at(y1, x0) * dy * (1 - dx) + at(y1, x1) * dy * dx; }
        report(std::string("GridSampler2D align=") + std::to_string(align), norm_err(hz, ref), 1e-5);
        aclDestroyTensor(ta); aclDestroyTensor(tg); aclDestroyTensor(tz);
    }
    for (int align = 0; align <= 1; align++) {
        int N = 1, C = 1, D = 3, H = 3, W = 3, oD = 2, oH = 2, oW = 2; auto v = randv(N * C * D * H * W, -2, 2); auto grid = randv(N * oD * oH * oW * 3, -1, 1);
        std::vector<float> hz(N * C * oD * oH * oW); DevBuf da(N * C * D * H * W * 4), dg(N * oD * oH * oW * 3 * 4), dz(N * C * oD * oH * oW * 4); da.up(v.data()); dg.up(grid.data());
        aclTensor *ta = mk({N, C, D, H, W}, ACL_FLOAT, da.p), *tg = mk({N, oD, oH, oW, 3}, ACL_FLOAT, dg.p), *tz = mk({N, C, oD, oH, oW}, ACL_FLOAT, dz.p);
        exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnGridSampler3DGetWorkspaceSize(ta, tg, 0, 0, (bool)align, tz, w, e); }, aclnnGridSampler3D);
        dz.down(hz.data()); std::vector<double> ref(N * C * oD * oH * oW);
        for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) for (int od = 0; od < oD; od++) for (int oh = 0; oh < oH; oh++) for (int ow = 0; ow < oW; ow++) {
            const float *gg = &grid[(((n * oD + od) * oH + oh) * oW + ow) * 3]; float gx = gg[0], gy = gg[1], gz = gg[2], fx, fy, fz;
            if (align) { fx = (gx + 1) * 0.5f * (W - 1); fy = (gy + 1) * 0.5f * (H - 1); fz = (gz + 1) * 0.5f * (D - 1); }
            else { fx = ((gx + 1) * W - 1) * 0.5f; fy = ((gy + 1) * H - 1) * 0.5f; fz = ((gz + 1) * D - 1) * 0.5f; }
            int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy), z0 = (int)std::floor(fz); float dx = fx - x0, dy = fy - y0, dz_ = fz - z0; const float *p = &v[(n * C + c) * D * H * W];
            auto at = [&](int z, int y, int x) -> double { return (z >= 0 && z < D && y >= 0 && y < H && x >= 0 && x < W) ? p[(z * H + y) * W + x] : 0.0; };
            double acc = 0; for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int cc = 0; cc < 2; cc++) {
                double wz = a ? dz_ : 1 - dz_, wy = b ? dy : 1 - dy, wx = cc ? dx : 1 - dx; acc += wz * wy * wx * at(z0 + a, y0 + b, x0 + cc); }
            ref[(((n * C + c) * oD + od) * oH + oh) * oW + ow] = acc; }
        report(std::string("GridSampler3D align=") + std::to_string(align), norm_err(hz, ref), 1e-5);
        aclDestroyTensor(ta); aclDestroyTensor(tg); aclDestroyTensor(tz);
    }
}

static void t_conv() {
    // ConvDepthwise2d: in[1,3,5,5], weight[3,1,3,3], bias[3], s=1 p=1 -> out[1,3,5,5]
    { int C = 3, H = 5, W = 5, R = 3, S = 3, s = 1, p = 1; int Ho = (H + 2 * p - R) / s + 1, Wo = (W + 2 * p - S) / s + 1;
      auto x = randv(C * H * W, -1, 1); auto wt = randv(C * 1 * R * S, -1, 1); auto bs = randv(C, -1, 1);
      std::vector<float> hz(C * Ho * Wo); DevBuf dx(C * H * W * 4), dw(C * R * S * 4), db(C * 4), dz(C * Ho * Wo * 4);
      dx.up(x.data()); dw.up(wt.data()); db.up(bs.data());
      aclTensor *tx = mk({1, C, H, W}, ACL_FLOAT, dx.p), *tw = mk({C, 1, R, S}, ACL_FLOAT, dw.p), *tb = mk({C}, ACL_FLOAT, db.p), *tz = mk({1, C, Ho, Wo}, ACL_FLOAT, dz.p);
      aclIntArray *ks = ia({R, S}), *str = ia({s, s}), *pd = ia({p, p}), *dl = ia({1, 1});
      exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnConvDepthwise2dGetWorkspaceSize(tx, tw, ks, tb, str, pd, dl, tz, w, e); }, aclnnConvDepthwise2d);
      dz.down(hz.data()); std::vector<double> ref(C * Ho * Wo);
      for (int c = 0; c < C; c++) for (int ho = 0; ho < Ho; ho++) for (int wo = 0; wo < Wo; wo++) {
          double acc = bs[c]; for (int kr = 0; kr < R; kr++) for (int ksx = 0; ksx < S; ksx++) { int hi = ho * s - p + kr, wi = wo * s - p + ksx;
              if (hi < 0 || hi >= H || wi < 0 || wi >= W) continue; acc += (double)x[(c * H + hi) * W + wi] * wt[(c * R + kr) * S + ksx]; }
          ref[(c * Ho + ho) * Wo + wo] = acc; }
      report("ConvDepthwise2d", norm_err(hz, ref), 1e-5); aclDestroyTensor(tx); aclDestroyTensor(tw); aclDestroyTensor(tb); aclDestroyTensor(tz);
      aclDestroyIntArray(ks); aclDestroyIntArray(str); aclDestroyIntArray(pd); aclDestroyIntArray(dl); }

    // FusedCausalConv1d: x[2,3,7], weight[3,4], bias[3]; act 0 (linear) and 1 (silu)
    for (int act = 0; act <= 1; act++) {
        int B = 2, C = 3, L = 7, K = 4; auto x = randv(B * C * L, -1, 1); auto wt = randv(C * K, -1, 1); auto bs = randv(C, -1, 1);
        std::vector<float> hz(B * C * L); DevBuf dx(B * C * L * 4), dw(C * K * 4), db(C * 4), dz(B * C * L * 4); dx.up(x.data()); dw.up(wt.data()); db.up(bs.data());
        aclTensor *tx = mk({B, C, L}, ACL_FLOAT, dx.p), *tw = mk({C, K}, ACL_FLOAT, dw.p), *tb = mk({C}, ACL_FLOAT, db.p), *tz = mk({B, C, L}, ACL_FLOAT, dz.p);
        exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnFusedCausalConv1dGetWorkspaceSize(tx, tw, tb, act, tz, w, e); }, aclnnFusedCausalConv1d);
        dz.down(hz.data()); std::vector<double> ref(B * C * L);
        for (int b = 0; b < B; b++) for (int c = 0; c < C; c++) for (int t = 0; t < L; t++) {
            double acc = bs[c]; for (int k = 0; k < K; k++) { int ti = t - (K - 1) + k; if (ti >= 0) acc += (double)wt[c * K + k] * x[(b * C + c) * L + ti]; }
            ref[(b * C + c) * L + t] = act == 1 ? acc / (1.0 + std::exp(-acc)) : acc; }
        report(std::string("FusedCausalConv1d act=") + std::to_string(act), norm_err(hz, ref), 1e-5);
        aclDestroyTensor(tx); aclDestroyTensor(tw); aclDestroyTensor(tb); aclDestroyTensor(tz);
    }

    // TransConvolutionWeight: pure copy
    { int n = 24; auto wt = randv(n, -3, 3); std::vector<float> hz(n); DevBuf dw(n * 4), dz(n * 4); dw.up(wt.data());
      aclTensor *tw = mk({2, 3, 2, 2}, ACL_FLOAT, dw.p), *tz = mk({2, 3, 2, 2}, ACL_FLOAT, dz.p);
      exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnTransConvolutionWeightGetWorkspaceSize(tw, tz, w, e); }, aclnnTransConvolutionWeight);
      dz.down(hz.data()); double bad = 0; for (int i = 0; i < n; i++) bad = std::max(bad, (double)std::fabs(hz[i] - wt[i]));
      report("TransConvolutionWeight", bad, 0); aclDestroyTensor(tw); aclDestroyTensor(tz); }
}

static void t_msda() {
    // MultiScaleDeformableAttnFunction: small single-level case.
    // value[N,S,nH,hd]; shapes[L,2]; lstart[L]; samp[N,Lq,nH,L,P,2]; attn[N,Lq,nH,L,P] -> out[N,Lq,nH,hd]
    int N = 1, nH = 2, hd = 3, Lq = 2, L = 1, P = 2; int Hs = 3, Ws = 4; int S = Hs * Ws;
    auto value = randv(N * S * nH * hd, -1, 1);
    int64_t shapes[2] = {Hs, Ws}; int64_t lstart[1] = {0};
    auto samp = randv(N * Lq * nH * L * P * 2, 0.05f, 0.95f);   // normalized [0,1]
    auto attn = randv(N * Lq * nH * L * P, 0.1f, 1.0f);
    std::vector<float> hz(N * Lq * nH * hd);
    DevBuf dv(value.size() * 4), dsh(2 * 8), dls(1 * 8), dsa(samp.size() * 4), dat(attn.size() * 4), dz(hz.size() * 4);
    dv.up(value.data()); dsh.up(shapes); dls.up(lstart); dsa.up(samp.data()); dat.up(attn.data());
    aclTensor *tv = mk({N, S, nH, hd}, ACL_FLOAT, dv.p), *tsh = mk({L, 2}, ACL_INT64, dsh.p), *tls = mk({L}, ACL_INT64, dls.p),
              *tsa = mk({N, Lq, nH, L, P, 2}, ACL_FLOAT, dsa.p), *tat = mk({N, Lq, nH, L, P}, ACL_FLOAT, dat.p), *tz = mk({N, Lq, nH, hd}, ACL_FLOAT, dz.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize(tv, tsh, tls, tsa, tat, tz, w, e); }, aclnnMultiScaleDeformableAttnFunction);
    dz.down(hz.data()); std::vector<double> ref(N * Lq * nH * hd, 0);
    for (int n = 0; n < N; n++) for (int q = 0; q < Lq; q++) for (int h = 0; h < nH; h++) for (int d = 0; d < hd; d++) {
        double acc = 0; for (int l = 0; l < L; l++) { int H = (int)shapes[l * 2], W = (int)shapes[l * 2 + 1]; int64_t base = lstart[l];
            for (int p = 0; p < P; p++) { int64_t si = ((((int64_t)n * Lq + q) * nH + h) * L + l) * P + p;
                float x = samp[si * 2] * W - 0.5f, y = samp[si * 2 + 1] * H - 0.5f, aw = attn[si];
                int x0 = (int)std::floor(x), y0 = (int)std::floor(y), x1 = x0 + 1, y1 = y0 + 1; float dx = x - x0, dy = y - y0;
                auto val = [&](int yy, int xx) -> double { if (yy < 0 || yy >= H || xx < 0 || xx >= W) return 0.0; int64_t sidx = base + (int64_t)yy * W + xx;
                    return value[(((int64_t)n * S + sidx) * nH + h) * hd + d]; };
                double v = (1 - dy) * (1 - dx) * val(y0, x0) + (1 - dy) * dx * val(y0, x1) + dy * (1 - dx) * val(y1, x0) + dy * dx * val(y1, x1);
                acc += aw * v; } }
        ref[(((int64_t)n * Lq + q) * nH + h) * hd + d] = acc; }
    report("MultiScaleDeformableAttnFunction", norm_err(hz, ref), 1e-5);
    aclDestroyTensor(tv); aclDestroyTensor(tsh); aclDestroyTensor(tls); aclDestroyTensor(tsa); aclDestroyTensor(tat); aclDestroyTensor(tz);
}

int main() {
    srand(1234); init();
    t_upsample_nearest();   // Nearest1dV2, Nearest2dV2, Nearest3d, NearestExact1d/2d/3d
    t_upsample_linear();    // Linear1d, Trilinear3d
    t_upsample_2d();        // Bilinear2D, Bilinear2dAA, Bicubic2dAA
    t_upsample_bwd();       // Bilinear2dBackwardV2
    t_pool();               // GlobalMaxPool, MaxPool, MaxPool2dWithMask
    t_roi();                // RoiAlignV2, RoiPoolingGradWithArgMax
    t_gridsampler();        // GridSampler2D, GridSampler3D
    t_conv();               // ConvDepthwise2d, FusedCausalConv1d, TransConvolutionWeight
    t_msda();               // MultiScaleDeformableAttnFunction
    return finish();
}
