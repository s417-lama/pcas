#pragma once

#include <cassert>
#include <type_traits>

#include <mpi.h>

#include "doctest/doctest.h"

#include "pcas/util.hpp"
#include "pcas/global_ptr.hpp"
#include "pcas/physical_mem.hpp"
#include "pcas/virtual_mem.hpp"

namespace pcas {

enum class dist_policy {
  local,
  block,
  block_cyclic,
};

using obj_id_t = uint64_t;

struct obj_entry {
  int          owner;
  obj_id_t     id;
  uint64_t     size;
  dist_policy  dpolicy;
  uint64_t     block_size;
  physical_mem pm;
  virtual_mem  vm;
  MPI_Win      win;
};

enum class access_mode {
  read,
  write,
  read_write,
};

struct checkout_entry {
  global_ptr<uint8_t> ptr;
  uint8_t*            raw_ptr;
  uint64_t            size;
  access_mode         mode;
};

class pcas {
  int rank_;
  int nproc_;
  MPI_Comm comm_;

  obj_id_t obj_id_count_ = 0; // TODO: better management of used IDs
  std::unordered_map<obj_id_t, obj_entry> objs_;

  std::unordered_map<void*, checkout_entry> checkouts_;

public:
  pcas(MPI_Comm comm = MPI_COMM_WORLD);
  ~pcas();

  int rank() const { return rank_; }
  int nproc() const { return nproc_; }

  void barrier() const { MPI_Barrier(comm_); }

  template <typename T>
  global_ptr<T> malloc(uint64_t nelems,
                       dist_policy dpolicy = dist_policy::block,
                       uint64_t block_size = 0);

  template <typename T>
  void free(global_ptr<T> ptr);

  template <typename T, typename Func>
  void for_each_block(global_ptr<T> ptr, uint64_t nelems, Func fn);

  template <typename T>
  void get(global_ptr<T> from_ptr, T* to_ptr, uint64_t nelems);

  template <typename T>
  void put(const T* from_ptr, global_ptr<T> to_ptr, uint64_t nelems);

  template <access_mode Mode, typename T>
  std::conditional_t<Mode == access_mode::read, const T*, T*>
  checkout(global_ptr<T> ptr, uint64_t nelems);

  template <typename T>
  void checkin(T* raw_ptr);

  template <typename T>
  void checkin(const T* raw_ptr);
};

pcas::pcas(MPI_Comm comm) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    die("MPI_Init() must be called before initializing PCAS.");
  }

  comm_ = comm;

  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &nproc_);

  barrier();
}

pcas::~pcas() {
  barrier();
}

TEST_CASE("initialize and finalize PCAS") {
  for (int i = 0; i < 3; i++) {
    pcas pc;
  }
}

template <typename T>
global_ptr<T> pcas::malloc(uint64_t nelems,
                           dist_policy dpolicy,
                           uint64_t block_size) {
  if (nelems == 0) {
    die("nelems cannot be 0");
  }

  switch (dpolicy) {
    case dist_policy::block: {
      uint64_t size_bytes = nelems * sizeof(T);
      uint64_t local_size = local_block_size(size_bytes, nproc_);

      virtual_mem vm(nullptr, local_size * nproc_);
      physical_mem pm(local_size);

      void* vm_local_addr = vm.map_physical_mem(rank_ * local_size, 0, local_size, pm);

      MPI_Win win = MPI_WIN_NULL;
      MPI_Win_create(vm_local_addr,
                     local_size,
                     1,
                     MPI_INFO_NULL,
                     comm_,
                     &win);
      MPI_Win_lock_all(0, win);

      obj_entry entry {
        .owner = -1, .id = obj_id_count_++, .size = size_bytes,
        .dpolicy = dpolicy, .block_size = local_size,
        .pm = std::move(pm), .vm = std::move(vm), .win = win,
      };

      auto ret = global_ptr<T>(entry.owner, entry.id, 0);

      objs_[entry.id] = std::move(entry);

      return ret;
    }
    default: {
      die("unimplemented");
      return global_ptr<T>();
    }
  }
}

template <typename T>
void pcas::free(global_ptr<T> ptr) {
  if (ptr == global_ptr<T>()) {
    die("null pointer was passed to pcas::free()");
  }
  obj_entry& entry = objs_[ptr.id()];
  switch (entry.dpolicy) {
    case dist_policy::block: {
      MPI_Win_unlock_all(entry.win);
      MPI_Win_free(&entry.win);
      break;
    }
    default: {
      die("unimplemented");
      break;
    }
  }
  objs_.erase(ptr.id());
}

TEST_CASE("malloc and free with block policy") {
  pcas pc;
  int n = 10;
  SUBCASE("free immediately") {
    for (int i = 1; i < n; i++) {
      auto p = pc.malloc<int>(i * 1234);
      pc.free(p);
    }
  }
  SUBCASE("free after accumulation") {
    global_ptr<int> ptrs[n];
    for (int i = 1; i < n; i++) {
      ptrs[i] = pc.malloc<int>(i * 2743);
    }
    for (int i = 1; i < n; i++) {
      pc.free(ptrs[i]);
    }
  }
}

template <typename T, typename Func>
void pcas::for_each_block(global_ptr<T> ptr, uint64_t nelems, Func fn) {
  obj_entry& entry = objs_[ptr.id()];

  uint64_t offset_min = ptr.offset();
  uint64_t offset_max = offset_min + nelems * sizeof(T);
  uint64_t offset     = offset_min;

  assert(offset_max <= entry.size);

  std::vector<MPI_Request> reqs;
  while (offset < offset_max) {
    auto [owner, idx_b, idx_e] = block_index_info(offset, entry.size, nproc_);
    uint64_t ib = std::max(idx_b, offset_min);
    uint64_t ie = std::min(idx_e, offset_max);

    fn(owner, ib, ie);

    offset = idx_e;
  }
}

TEST_CASE("loop over blocks") {
  pcas pc;

  int nproc = pc.nproc();

  int n = 100000;
  auto p = pc.malloc<int>(n);

  SUBCASE("loop over the entire array") {
    int prev_owner = -1;
    int prev_ie = 0;
    pc.for_each_block(p, n, [&](int owner, uint64_t ib, uint64_t ie) {
      CHECK(owner == prev_owner + 1);
      CHECK(ib == prev_ie);
      prev_owner = owner;
      prev_ie = ie;
    });
    CHECK(prev_owner == nproc - 1);
    CHECK(prev_ie == n * sizeof(int));
  }

  SUBCASE("loop over the partial array") {
    int b = n / 5 * 2;
    int e = n / 5 * 4;
    int s = e - b;

    auto [o1, _ib1, _ie1] = block_index_info(b * sizeof(int), n * sizeof(int), nproc);
    auto [o2, _ib2, _ie2] = block_index_info(e * sizeof(int), n * sizeof(int), nproc);

    int prev_owner = o1 - 1;
    int prev_ie = b * sizeof(int);
    pc.for_each_block(p + b, s, [&](int owner, uint64_t ib, uint64_t ie) {
      CHECK(owner == prev_owner + 1);
      CHECK(ib == prev_ie);
      prev_owner = owner;
      prev_ie = ie;
    });
    CHECK(prev_owner == o2);
    CHECK(prev_ie == e * sizeof(int));
  }

  pc.free(p);
}

template <typename T>
void pcas::get(global_ptr<T> from_ptr, T* to_ptr, uint64_t nelems) {
  if (from_ptr.owner() == -1) {
    obj_entry& entry = objs_[from_ptr.id()];
    uint64_t offset = from_ptr.offset();
    std::vector<MPI_Request> reqs;

    for_each_block(from_ptr, nelems, [&](int owner, uint64_t idx_b, uint64_t idx_e) {
      MPI_Request req;
      MPI_Rget((uint8_t*)to_ptr - offset + idx_b,
               idx_e - idx_b,
               MPI_UINT8_T,
               owner,
               idx_b - owner * entry.block_size,
               idx_e - idx_b,
               MPI_UINT8_T,
               entry.win,
               &req);
      reqs.push_back(req);
    });

    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
  } else {
    die("unimplemented");
  }
}

template <typename T>
void pcas::put(const T* from_ptr, global_ptr<T> to_ptr, uint64_t nelems) {
  if (to_ptr.owner() == -1) {
    obj_entry& entry = objs_[to_ptr.id()];
    uint64_t offset = to_ptr.offset();
    std::vector<MPI_Request> reqs;

    for_each_block(to_ptr, nelems, [&](int owner, uint64_t idx_b, uint64_t idx_e) {
      MPI_Request req;
      MPI_Rput((uint8_t*)from_ptr - offset + idx_b,
               idx_e - idx_b,
               MPI_UINT8_T,
               owner,
               idx_b - owner * entry.block_size,
               idx_e - idx_b,
               MPI_UINT8_T,
               entry.win,
               &req);
      reqs.push_back(req);
    });

    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    // ensure remote completion
    MPI_Win_flush_all(entry.win);
  } else {
    die("unimplemented");
  }
}

TEST_CASE("get and put") {
  pcas pc;

  int rank = pc.rank();

  int n = 100000;
  auto p = pc.malloc<int>(n);

  int* buf = new int[n + 2];

  if (rank == 0) {
    for (int i = 0; i < n; i++) {
      buf[i] = i;
    }
    pc.put(buf, p, n);
  }

  pc.barrier();

  SUBCASE("get the entire array") {
    int special = 417;
    buf[0] = buf[n + 1] = special;

    pc.get(p, buf + 1, n);

    for (int i = 0; i < n; i++) {
      CHECK(buf[i + 1] == i);
    }
    CHECK(buf[0]     == special);
    CHECK(buf[n + 1] == special);
  }

  SUBCASE("get the partial array") {
    int ib = n / 5 * 2;
    int ie = n / 5 * 4;
    int s = ie - ib;

    int special = 417;
    buf[0] = buf[s + 1] = special;

    pc.get(p + ib, buf + 1, s);

    for (int i = 0; i < s; i++) {
      CHECK(buf[i + 1] == i + ib);
    }
    CHECK(buf[0]     == special);
    CHECK(buf[s + 1] == special);
  }

  SUBCASE("get each element") {
    for (int i = 0; i < n; i++) {
      int special = 417;
      buf[0] = buf[2] = special;
      pc.get(p + i, &buf[1], 1);
      CHECK(buf[0] == special);
      CHECK(buf[1] == i);
      CHECK(buf[2] == special);
    }
  }

  pc.free(p);
}

template <access_mode Mode, typename T>
std::conditional_t<Mode == access_mode::read, const T*, T*>
pcas::checkout(global_ptr<T> ptr, uint64_t nelems) {
  T* ret = (T*)std::malloc(nelems * sizeof(T));
  if (Mode == access_mode::read_write ||
      Mode == access_mode::read) {
    get(ptr, ret, nelems);
  }
  checkouts_[(void*)ret] = (checkout_entry){
    .ptr = static_cast<global_ptr<uint8_t>>(ptr), .raw_ptr = (uint8_t*)ret,
    .size = nelems * sizeof(T), .mode = Mode,
  };
  return ret;
}

template <typename T>
void pcas::checkin(T* raw_ptr) {
  auto c = checkouts_.find((void*)raw_ptr);
  if (c == checkouts_.end()) {
    die("The pointer %p passed to checkin() is not registered", raw_ptr);
  }
  checkout_entry entry = c->second;
  if (entry.mode == access_mode::read_write ||
      entry.mode == access_mode::write) {
    put(entry.raw_ptr, entry.ptr, entry.size);
  }
  std::free(raw_ptr);
  checkouts_.erase((void*)raw_ptr);
}

template <typename T>
void pcas::checkin(const T* raw_ptr) {
  checkin(const_cast<T*>(raw_ptr));
}

TEST_CASE("checkout and checkin") {
  pcas pc;

  int rank = pc.rank();
  int nproc = pc.nproc();

  int n = 100000;
  auto p = pc.malloc<int>(n);

  if (rank == 0) {
    int* rp = pc.checkout<access_mode::write>(p, n);
    for (int i = 0; i < n; i++) {
      rp[i] = i;
    }
    pc.checkin(rp);
  }

  pc.barrier();

  SUBCASE("read the entire array") {
    const int* rp = pc.checkout<access_mode::read>(p, n);
    for (int i = 0; i < n; i++) {
      CHECK(rp[i] == i);
    }
    pc.checkin(rp);
  }

  SUBCASE("read the partial array") {
    int ib = n / 5 * 2;
    int ie = n / 5 * 4;
    int s = ie - ib;

    const int* rp = pc.checkout<access_mode::read>(p + ib, s);
    for (int i = 0; i < s; i++) {
      CHECK(rp[i] == i + ib);
    }
    pc.checkin(rp);
  }

  SUBCASE("read and write the partial array") {
    int stride = 48;
    for (int i = rank * stride; i < n; i += nproc * stride) {
      int s = std::min(stride, n - i);
      int* rp = pc.checkout<access_mode::read_write>(p + i, s);
      for (int j = 0; j < s; j++) {
        rp[j] *= 2;
      }
      pc.checkin(rp);
    }

    pc.barrier();

    const int* rp = pc.checkout<access_mode::read>(p, n);
    for (int i = 0; i < n; i++) {
      CHECK(rp[i] == i * 2);
    }
    pc.checkin(rp);
  }

  pc.free(p);
}

}
