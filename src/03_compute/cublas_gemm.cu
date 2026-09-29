// Phase 3 (算力密度基线): cuBLAS 实际可达算力——Phase 5 综合 GEMM 的对标线。
// fp16: GemmEx compute-32F (张量核) 8192^3
// fp32: Sgemm 4096^3
// 先做 64^3 fp16 小尺寸正确性 sanity（防 API 用法错误）。
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "common/bench.h"
#include "common/checker.h"
#include "common/device_info.h"
#include "common/errors.h"
#include "common/tensor.h"
#include "common/timer.h"

#define CUBLAS_CHECK(cmd)                                                    \
  do {                                                                       \
    cublasStatus_t s_ = (cmd);                                               \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "cuBLAS error at %s:%d: status=%d (%s)\n",        \
                   __FILE__, __LINE__, (int)s_, #cmd);                       \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

namespace {

// 列主序 C = A * B (N,N)，CPU float 参考实现
std::vector<float> cpu_gemm_f32(const std::vector<__half>& a,
                                const std::vector<__half>& b, uint32_t n) {
  std::vector<float> c(size_t(n) * n);
  for (uint32_t j = 0; j < n; ++j)
    for (uint32_t i = 0; i < n; ++i) {
      float acc = 0.f;
      for (uint32_t k = 0; k < n; ++k) {
        acc += float(a[size_t(i) + size_t(k) * n]) * float(b[size_t(k) + size_t(j) * n]);
      }
      c[size_t(i) + size_t(j) * n] = acc;
    }
  return c;
}

}  // namespace

int main() {
  auto caps = hopper::query_device();
  hopper::print_device_caps(caps);
  hopper::require_hopper(caps);

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  const float alpha = 1.f, beta = 0.f;

  // ---- 正确性 sanity: 64^3 fp16, compute-32F ----
  {
    constexpr uint32_t n = 64;
    auto fa = hopper::make_random_vector(size_t(n) * n, 101, -1.f, 1.f);
    auto fb = hopper::make_random_vector(size_t(n) * n, 102, -1.f, 1.f);
    std::vector<__half> ha(fa.begin(), fa.end()), hb(fb.begin(), fb.end());
    hopper::DeviceBuffer<__half> da(size_t(n) * n), db(size_t(n) * n), dc(size_t(n) * n);
    da.upload(ha);
    db.upload(hb);
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                              da.get(), CUDA_R_16F, n, db.get(), CUDA_R_16F, n,
                              &beta, dc.get(), CUDA_R_16F, n,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    auto hc = dc.download();
    std::vector<float> got(hc.begin(), hc.end());
    auto ref = cpu_gemm_f32(ha, hb, n);
    auto rep = hopper::check_close(got, ref, 1e-2, 1e-2);  // fp16 输出 + 求和序差异
    hopper::print_report(rep, "cublas hgemm 64^3");
    if (!rep.pass) return 1;
  }

  struct Result {
    const char* name;
    double tflops;
  };
  std::vector<Result> results;

  // ---- fp16 tensor: 8192^3 ----
  {
    constexpr uint32_t n = 8192;
    auto fa = hopper::make_random_vector(size_t(n) * n, 103, -1.f, 1.f);
    auto fb = hopper::make_random_vector(size_t(n) * n, 104, -1.f, 1.f);
    std::vector<__half> ha(fa.begin(), fa.end()), hb(fb.begin(), fb.end());
    hopper::DeviceBuffer<__half> da(size_t(n) * n), db(size_t(n) * n), dc(size_t(n) * n);
    da.upload(ha);
    db.upload(hb);
    auto st = hopper::time_reps(5, 20, [&] {
      CUBLAS_CHECK(cublasGemmEx(
          handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, da.get(),
          CUDA_R_16F, n, db.get(), CUDA_R_16F, n, &beta, dc.get(), CUDA_R_16F,
          n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    });
    results.push_back({"cuBLAS HGEMM 8192^3 (fp16->fp16, tc)",
                       hopper::tflops(2.0 * n * n * n, st.min_ms)});
  }

  // ---- fp32: 4096^3 ----
  {
    constexpr uint32_t n = 4096;
    auto ha = hopper::make_random_vector(size_t(n) * n, 105);
    auto hb = hopper::make_random_vector(size_t(n) * n, 106);
    hopper::DeviceBuffer<float> da(size_t(n) * n), db(size_t(n) * n),
        dc(size_t(n) * n);
    da.upload(ha);
    db.upload(hb);
    auto st = hopper::time_reps(5, 20, [&] {
      CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                               &alpha, da.get(), n, db.get(), n, &beta,
                               dc.get(), n));
    });
    results.push_back({"cuBLAS SGEMM 4096^3 (fp32)",
                       hopper::tflops(2.0 * n * n * n, st.min_ms)});
  }

  std::printf("\n[CUBLAS] 实际可达算力（满载 GPU 下偏低，Phase 5 对标线）\n");
  for (auto& r : results) {
    std::printf("  %-38s : %8.2f TFLOPS\n", r.name, r.tflops);
  }

  std::filesystem::create_directories("results");
  hopper::CsvWriter csv("results/cublas_gemm.csv", {"case", "tflops"});
  for (auto& r : results) csv.row({r.name, std::to_string(r.tflops)});
  return 0;
}
