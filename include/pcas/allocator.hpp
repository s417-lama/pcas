#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory_resource>
#include <unordered_map>
#include <mpi.h>

#include "pcas/util.hpp"
#include "pcas/logger/logger.hpp"

namespace pcas {

template <typename P>
class allocator_if final : public std::pmr::memory_resource {
protected:
  struct dynamic_win {
    MPI_Win win;
    dynamic_win(MPI_Comm comm) {
      MPI_Win_create_dynamic(MPI_INFO_NULL, comm, &win);
      MPI_Win_lock_all(0, win);
    }
    ~dynamic_win() {
      MPI_Win_unlock_all(win);
      MPI_Win_free(&win);
    }
  };

  dynamic_win dwin_;
  typename P::template allocator_impl_t<P> allocator_;

public:
  allocator_if(MPI_Comm comm) : dwin_(comm), allocator_(dwin_.win) {}

  MPI_Win get_win() { return dwin_.win; }

  void* do_allocate(std::size_t bytes, std::size_t alignment = alignof(max_align_t)) override {
    return allocator_.do_allocate(bytes, alignment);
  }

  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment = alignof(max_align_t)) override {
    allocator_.do_deallocate(p, bytes, alignment);
  }

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }
};

class mpi_win_resource final : public std::pmr::memory_resource {
  MPI_Win win_;
  std::unordered_map<void*, void*> allocated_;

public:
  mpi_win_resource(MPI_Win win) : win_(win) {}

  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    std::size_t bytes_pad = bytes + alignment;
    void* p;
    if (MPI_Alloc_mem(bytes_pad, MPI_INFO_NULL, &p) != MPI_SUCCESS) {
      throw std::bad_alloc{};
    }
    MPI_Win_attach(win_, p, bytes_pad);
    void* ret = (void*)(((uintptr_t)p + alignment - 1) / alignment * alignment);
    allocated_[ret] = p;
    return ret;
  }

  void do_deallocate(void* p, [[maybe_unused]] std::size_t bytes, [[maybe_unused]] std::size_t alignment) override {
    void* orig_p = allocated_[p];
    allocated_.erase(p);
    MPI_Win_detach(win_, orig_p);
    MPI_Free_mem(orig_p);
  }

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }
};

template <typename P>
class std_pool_resource_impl {
  mpi_win_resource win_mr_;
  std::pmr::unsynchronized_pool_resource mr_;

  using logger = typename P::logger;
  using logger_kind = typename P::logger::kind::value;

public:
  std_pool_resource_impl(MPI_Win win) :
    win_mr_(win),
    mr_(std::pmr::pool_options{.max_blocks_per_chunk = (std::size_t)16 * 1024 * 1024 * 1024}, &win_mr_) {
  }

  void* do_allocate(std::size_t bytes, std::size_t alignment) {
    auto ev = logger::template record<logger_kind::MemAlloc>(bytes);
    return mr_.allocate(bytes, alignment);
  }

  void do_deallocate(void* p, std::size_t bytes, [[maybe_unused]] std::size_t alignment) {
    auto ev = logger::template record<logger_kind::MemFree>(bytes);
    mr_.deallocate(p, bytes, alignment);
  }
};

struct allocator_policy_default {
  using logger = logger::logger_if<logger::policy_default>;
  template <typename P>
  using allocator_impl_t = std_pool_resource_impl<P>;
};

PCAS_TEST_CASE("[pcas::allocator] basic test") {
  allocator_if<allocator_policy_default> allocator(MPI_COMM_WORLD);

  PCAS_SUBCASE("Local alloc/dealloc") {
    std::vector<std::size_t> sizes = {1, 2, 4, 8, 16, 32, 100, 200, 1000, 100000, 1000000};
    constexpr int N = 10;
    for (auto size : sizes) {
      void* ptrs[N];
      for (int i = 0; i < N; i++) {
        ptrs[i] = allocator.allocate(size);
        for (std::size_t j = 0; j < size; j += 128) {
          ((char*)ptrs[i])[j] = 0;
        }
      }
      for (int i = 0; i < N; i++) {
        allocator.deallocate(ptrs[i], size);
      }
    }
  }

  PCAS_SUBCASE("Remote access") {
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    std::size_t size = 128;
    void* p = allocator.allocate(size);

    for (std::size_t i = 0; i < size; i++) {
      ((uint8_t*)p)[i] = rank;
    }

    std::vector<uint64_t> addrs(nproc);
    addrs[rank] = (uint64_t)p;

    // GET
    for (int target_rank = 0; target_rank < nproc; target_rank++) {
      MPI_Bcast(&addrs[target_rank], 1, MPI_UINT64_T, target_rank, MPI_COMM_WORLD);
      if (rank != target_rank) {
        std::vector<uint8_t> buf(size);
        MPI_Get(buf.data(),
                size,
                MPI_UINT8_T,
                target_rank,
                addrs[target_rank],
                size,
                MPI_UINT8_T,
                allocator.get_win());
        MPI_Win_flush(target_rank, allocator.get_win());

        for (std::size_t i = 0; i < size; i++) {
          PCAS_CHECK(buf[i] == target_rank);
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // PUT
    std::vector<uint8_t> buf(size);
    for (std::size_t i = 0; i < size; i++) {
      buf[i] = rank;
    }

    int target_rank = (rank + 1) % nproc;
    MPI_Put(buf.data(),
            size,
            MPI_UINT8_T,
            target_rank,
            addrs[target_rank],
            size,
            MPI_UINT8_T,
            allocator.get_win());
    MPI_Win_flush(target_rank, allocator.get_win());

    MPI_Barrier(MPI_COMM_WORLD);

    for (std::size_t i = 0; i < size; i++) {
      PCAS_CHECK(((uint8_t*)p)[i] == (nproc + rank - 1) % nproc);
    }
  }
}

}
