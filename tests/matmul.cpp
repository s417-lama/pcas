#include <cmath>
#include <vector>
#include <type_traits>
#include <random>

/* #define NDEBUG */
#include "pcas/pcas.hpp"

using real_t = double;
template <typename T>
using global_ptr = pcas::pcas::global_ptr<T>;

template <pcas::access_mode Mode, typename T>
auto checkout_stride(global_ptr<T> ptr,
                     uint64_t      bcount,
                     uint64_t      blen,
                     uint64_t      bstride,
                     pcas::pcas&   pc) {
  std::vector<std::conditional_t<Mode == pcas::access_mode::read, const T*, T*>> ret;
  for (uint64_t b = 0; b < bcount; b++) {
    ret.push_back(pc.checkout<Mode>(ptr + b * bstride, blen));
  }
  return ret;
}

template <typename T>
void checkin_stride(std::vector<T> vec,
                    uint64_t       blen,
                    pcas::pcas&    pc) {
  for (auto p : vec) {
    pc.checkin(p, blen);
  }
}

void matmul_seq(const real_t** A, const real_t** B, real_t** C, uint64_t n) {
  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t j = 0; j < n; j++) {
      for (uint64_t k = 0; k < n; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void matmul_par(global_ptr<real_t> A,
                global_ptr<real_t> B,
                global_ptr<real_t> C,
                uint64_t           n,
                uint64_t           bn,
                pcas::pcas&        pc) {
  for (uint64_t bi = 0; bi < n / bn; bi++) {
    for (uint64_t bj = 0; bj < n / bn; bj++) {
      if ((int)(bi * (n / bn) + bj) % pc.nproc() == pc.rank()) {
        auto C_ = checkout_stride<pcas::access_mode::read_write>(C + bi * bn * n + bj * bn, bn, bn, n, pc);
        for (uint64_t bk = 0; bk < n / bn; bk++) {
          auto A_ = checkout_stride<pcas::access_mode::read>(A + bi * bn * n + bk * bn, bn, bn, n, pc);
          auto B_ = checkout_stride<pcas::access_mode::read>(B + bk * bn * n + bj * bn, bn, bn, n, pc);

          matmul_seq(A_.data(), B_.data(), C_.data(), bn);

          checkin_stride(A_, bn, pc);
          checkin_stride(B_, bn, pc);
        }
        checkin_stride(C_, bn, pc);
      }
    }
  }
}

void matmul_init(global_ptr<real_t> A,
                 global_ptr<real_t> B,
                 global_ptr<real_t> C,
                 uint64_t           n,
                 pcas::pcas&        pc) {
  static std::mt19937 engine(0);
  std::uniform_real_distribution<> dist(-1.0, 1.0);

  constexpr uint64_t stride = 4096;
  for (uint64_t i = 0; i < n * n; i += stride) {
    uint64_t s = std::min(stride, n * n - i);
    auto A_ = pc.checkout<pcas::access_mode::write>(A + i, s);
    auto B_ = pc.checkout<pcas::access_mode::write>(B + i, s);
    auto C_ = pc.checkout<pcas::access_mode::write>(C + i, s);
    for (uint64_t j = 0; j < s; j++) {
      /* A_[j] = 1.0; */
      /* B_[j] = 1.0; */
      A_[j] = dist(engine);
      B_[j] = dist(engine);
      C_[j] = 0.0;
    }
    pc.checkin(A_, s);
    pc.checkin(B_, s);
    pc.checkin(C_, s);
  }
}

void matmul_check(global_ptr<real_t> A,
                  global_ptr<real_t> B,
                  global_ptr<real_t> C,
                  uint64_t           n,
                  pcas::pcas&        pc) {
  static std::mt19937 engine(0);
  std::uniform_int_distribution<> dist(0, n - 1);

  constexpr int ntrials = 100;

  for (int it = 0; it < ntrials; it++) {
    uint64_t i = dist(engine);
    uint64_t j = dist(engine);

    auto A_ = pc.checkout<pcas::access_mode::read>(A + i * n, n);
    real_t c_ans = 0;
    for (uint64_t k = 0; k < n; k++) {
      real_t b;
      pc.get(B + k * n + j, &b, 1);
      c_ans += A_[k] * b;
    }
    pc.checkin(A_, n);

    real_t c_computed;
    pc.get(C + i * n + j, &c_computed, 1);

    if (std::abs(c_ans - c_computed) > 1e-6) {
      pcas::die("C[%lu][%lu] Got: %f, Expected: %f\n", i, j, c_computed, c_ans);
    }
  }

  printf("check succeeded!\n");
}

void matmul_main(uint64_t n, uint64_t bn) {
  PCAS_CHECK(n % bn == 0);

  constexpr uint64_t cache_size = 16 * 1024 * 1024;
  pcas::pcas pc(cache_size);

  auto A = pc.malloc<real_t>(n * n);
  auto B = pc.malloc<real_t>(n * n);
  auto C = pc.malloc<real_t>(n * n);

  if (pc.rank() == 0)
    matmul_init(A, B, C, n, pc);

  pc.barrier();

  matmul_par(A, B, C, n, bn, pc);

  pc.barrier();

  if (pc.rank() == 0)
    matmul_check(A, B, C, n, pc);

  pc.free(A);
  pc.free(B);
  pc.free(C);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  matmul_main(500, 50);
  /* matmul_main(1000, 100); */

  MPI_Finalize();
  return 0;
}
