#pragma once

#include <cmath>
#include <cstdio>
#include <vector>

namespace hopper {

// 各 dtype 默认容差。fp16 / bf16 / fp8 特化在对应实验阶段补充。
template <typename T>
struct DefaultTol {
  static constexpr double rtol = 1e-5;
  static constexpr double atol = 1e-6;
};
template <>
struct DefaultTol<float> {
  static constexpr double rtol = 2e-5;
  static constexpr double atol = 1e-6;
};

struct CheckReport {
  bool pass = false;
  size_t n = 0;
  size_t mismatches = 0;
  double max_abs_err = 0.0;
  double max_rel_err = 0.0;
};

// 逐元素检查 |got - ref| <= atol + rtol * |ref|（与 torch.allclose 同款公式）
template <typename T>
CheckReport check_close(const std::vector<T>& got, const std::vector<T>& ref,
                        double rtol, double atol) {
  CheckReport r;
  r.n = got.size();
  if (got.size() != ref.size()) {
    r.mismatches = r.n;  // 尺寸不一致，全部计为失败
    return r;
  }
  for (size_t i = 0; i < r.n; ++i) {
    double g = static_cast<double>(got[i]);
    double e = static_cast<double>(ref[i]);
    double diff = std::fabs(g - e);
    double rel = diff / std::fmax(std::fabs(e), 1e-30);
    if (diff > r.max_abs_err) r.max_abs_err = diff;
    if (rel > r.max_rel_err) r.max_rel_err = rel;
    if (diff > atol + rtol * std::fabs(e)) ++r.mismatches;
  }
  r.pass = (r.mismatches == 0);
  return r;
}

template <typename T>
CheckReport check_close(const std::vector<T>& got, const std::vector<T>& ref) {
  return check_close(got, ref, DefaultTol<T>::rtol, DefaultTol<T>::atol);
}

inline void print_report(const CheckReport& r, const char* tag) {
  std::printf("[CHECK ] %-20s : %s  (n=%zu mismatch=%zu max_abs=%.3e max_rel=%.3e)\n",
              tag, r.pass ? "PASS" : "FAIL", r.n, r.mismatches,
              r.max_abs_err, r.max_rel_err);
}

}  // namespace hopper
