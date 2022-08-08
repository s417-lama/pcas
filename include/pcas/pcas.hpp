#pragma once

#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include <mpi.h>

#include "pcas/util.hpp"
#include "pcas/global_ptr.hpp"
#include "pcas/physical_mem.hpp"
#include "pcas/virtual_mem.hpp"
#include "pcas/cache.hpp"
#include "pcas/wallclock.hpp"
#include "pcas/logger/logger.hpp"

namespace pcas {

enum class dist_policy {
  local,
  block,
  block_cyclic,
};

using cache_t = cache_system<min_block_size>;

struct obj_entry {
  int                           owner;
  obj_id_t                      id;
  uint64_t                      size;
  uint64_t                      effective_size;
  dist_policy                   dpolicy;
  uint64_t                      block_size;
  std::vector<physical_mem>     pms;
  virtual_mem                   vm;
  std::vector<cache_t::entry_t> cache_entries;
  MPI_Win                       win;
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

using epoch_t = uint64_t;

struct release_remote_region {
  epoch_t request; // requested epoch from remote
  epoch_t epoch; // current epoch of the owner
};

struct release_manager {
  release_remote_region* remote;
  MPI_Win win;
};

struct release_handler {
  int rank;
  epoch_t epoch;
};

struct policy_default {
  using wallclock_t = wallclock_native;
  using logger_kind_t = logger::kind;
  template <typename P>
  using logger_impl_t = logger::impl_dummy<P>;
};

template <typename P>
class pcas_if;

using pcas = pcas_if<policy_default>;

template <typename P>
class pcas_if {
  int      global_rank_  = -1;
  int      global_nproc_ = -1;
  MPI_Comm global_comm_;

  int      intra_rank_  = -1;
  int      intra_nproc_ = -1;
  MPI_Comm intra_comm_;

  int      inter_rank_  = -1;
  int      inter_nproc_ = -1;
  MPI_Comm inter_comm_;

  std::vector<std::pair<int, int>> process_map_; // pair: (intra, inter rank)
  std::vector<int> intra2global_rank_;

  obj_id_t obj_id_count_ = 1; // TODO: better management of used IDs
  std::unordered_map<obj_id_t, obj_entry> objs_;

  std::unordered_map<void*, checkout_entry> checkouts_;

  cache_t cache_;

  bool cache_dirty_ = false;
  release_manager rm_;

  std::unordered_set<MPI_Win> flushing_wins_;

  uint64_t n_dirty_cache_blocks_ = 0;
  uint64_t max_dirty_cache_blocks_;

  std::vector<std::pair<int, int>> init_process_map(MPI_Comm comm) {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
      die("MPI_Init() must be called before initializing PCAS.");
    }

    global_comm_ = comm;
    MPI_Comm_rank(global_comm_, &global_rank_);
    MPI_Comm_size(global_comm_, &global_nproc_);

    MPI_Comm_split_type(global_comm_, MPI_COMM_TYPE_SHARED, global_rank_, MPI_INFO_NULL, &intra_comm_);
    MPI_Comm_rank(intra_comm_, &intra_rank_);
    MPI_Comm_size(intra_comm_, &intra_nproc_);

    MPI_Comm_split(global_comm_, intra_rank_, global_rank_, &inter_comm_);
    MPI_Comm_rank(inter_comm_, &inter_rank_);
    MPI_Comm_size(inter_comm_, &inter_nproc_);

    std::pair<int, int> myranks {intra_rank_, inter_rank_};
    std::pair<int, int>* buf = new std::pair<int, int>[global_nproc_];
    MPI_Allgather(&myranks,
                  sizeof(std::pair<int, int>),
                  MPI_BYTE,
                  buf,
                  sizeof(std::pair<int, int>),
                  MPI_BYTE,
                  global_comm_);

    return std::vector(buf, buf + global_nproc_);
  }

  std::vector<int> init_intra2global_rank() {
    std::vector<int> ret;
    for (int i = 0; i < global_nproc_; i++) {
      if (process_map_[i].second == inter_rank_) {
        ret.push_back(i);
      }
    }
    PCAS_CHECK(ret.size() == (size_t)intra_nproc_);
    return ret;
  }

public:
  using logger = typename logger::template logger_if<logger::policy<P>>;
  using logger_kind = typename P::logger_kind_t::value;

  pcas_if(uint64_t cache_size = 1024 * min_block_size, MPI_Comm comm = MPI_COMM_WORLD);
  ~pcas_if();

  int rank() const { return global_rank_; }
  int nproc() const { return global_nproc_; }

  void flush_dirty_cache() {
    if (cache_dirty_) {
      cache_.for_each_block([&](cache_t::entry_t cae) {
        if (cae && !cae->dirty_sections.empty() && cae->checkout_count == 0) {
          PCAS_CHECK(!cae->flushing);

          obj_entry& obe = objs_[cae->obj_id];
          void* cache_block_ptr = cache_.pm().anon_vm_addr();
          uint64_t vm_offset = cae->vm_addr - (uint8_t*)obe.vm.addr();
          auto [owner, _idx_b, _idx_e] =
            block_index_info(vm_offset, obe.effective_size, global_nproc_);

          for (auto [offset_in_block_b, offset_in_block_e] : cae->dirty_sections) {
            MPI_Put((uint8_t*)cache_block_ptr + cae->block_num * cache_t::block_size + offset_in_block_b,
                    offset_in_block_e - offset_in_block_b,
                    MPI_UINT8_T,
                    owner,
                    vm_offset - owner * obe.block_size + offset_in_block_b,
                    offset_in_block_e - offset_in_block_b,
                    MPI_UINT8_T,
                    obe.win);
          }
          cae->dirty_sections.clear();
          cae->flushing = true;

          flushing_wins_.insert(obe.win);
          n_dirty_cache_blocks_--;
        }
      });
    }
  }

  void complete_flush() {
    if (!flushing_wins_.empty()) {
      for (auto win : flushing_wins_) {
        MPI_Win_flush_all(win);
      }
      flushing_wins_.clear();

      cache_.for_each_block([&](cache_t::entry_t cae) {
        if (cae && cae->flushing) cae->flushing = false;
      });
    }
  }

  void ensure_all_cache_clean() {
    PCAS_CHECK(checkouts_.empty());
    flush_dirty_cache();
    complete_flush();
    PCAS_CHECK(n_dirty_cache_blocks_ == 0);

    cache_dirty_ = false;
    rm_.remote->epoch++;
  }

  void release() {
    auto ev = logger::template record<logger_kind::Release>();
    ensure_all_cache_clean();
  }

  void release_lazy(release_handler* handler) {
    PCAS_CHECK(checkouts_.empty());

    epoch_t next_epoch = cache_dirty_ ? rm_.remote->epoch + 1 : 0; // 0 means clean
    *handler = {.rank = global_rank_, .epoch = next_epoch};
  }

  epoch_t get_remote_epoch(int target_rank) {
    epoch_t remote_epoch;

    MPI_Request req;
    MPI_Rget(&remote_epoch,
             sizeof(epoch_t),
             MPI_UINT8_T,
             target_rank,
             offsetof(release_remote_region, epoch),
             sizeof(epoch_t),
             MPI_UINT8_T,
             rm_.win,
             &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);

    return remote_epoch;
  }

  void send_release_request(int target_rank, epoch_t request) {
    epoch_t prev;
    MPI_Fetch_and_op(&request,
                     &prev,
                     MPI_UINT64_T, // should match epoch_t
                     target_rank,
                     offsetof(release_remote_region, request),
                     MPI_MAX,
                     rm_.win);
    MPI_Win_flush(target_rank, rm_.win);
  }

  void acquire(release_handler handler = {.rank = 0, .epoch = 0}) {
    auto ev = logger::template record<logger_kind::Acquire>();
    ensure_all_cache_clean();

    if (handler.epoch != 0) {
      if (get_remote_epoch(handler.rank) < handler.epoch) {
        send_release_request(handler.rank, handler.epoch);
        // need to wait for the execution of a release by the remote worker
        while (get_remote_epoch(handler.rank) < handler.epoch) {
          usleep(10); // TODO: better interval?
        };
      }
    }

    cache_.evict_all();
  }

  void barrier() {
    release();
    MPI_Barrier(global_comm_);
    acquire();
  }

  void poll() {
    if (n_dirty_cache_blocks_ >= max_dirty_cache_blocks_) {
      auto ev = logger::template record<logger_kind::FlushEarly>();
      flush_dirty_cache();
    }

    if (rm_.remote->request > rm_.remote->epoch) {
      auto ev = logger::template record<logger_kind::ReleaseLazy>();
      PCAS_CHECK(rm_.remote->request == rm_.remote->epoch + 1);

      ensure_all_cache_clean();

      PCAS_CHECK(rm_.remote->request == rm_.remote->epoch);
    }
  }

  template <typename T>
  global_ptr<T> malloc(uint64_t    nelems,
                       dist_policy dpolicy    = dist_policy::block,
                       uint64_t    block_size = 0);

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

  /* unsafe APIs for debugging */

  template <typename T>
  void* get_physical_mem(global_ptr<T> ptr) {
    obj_entry& obe = objs_[ptr.id()];
    return obe.pms[intra_rank_].anon_vm_addr();
  }

};

template <typename P>
inline pcas_if<P>::pcas_if(uint64_t cache_size, MPI_Comm comm)
  : process_map_(init_process_map(comm)),
    intra2global_rank_(init_intra2global_rank()),
    cache_(cache_size, intra_rank_) {
  logger::init(global_rank_, global_nproc_);

  MPI_Win_allocate(sizeof(release_remote_region),
                   1,
                   MPI_INFO_NULL,
                   global_comm_,
                   &rm_.remote,
                   &rm_.win);
  MPI_Win_lock_all(0, rm_.win);

  rm_.remote->request = 1;
  rm_.remote->epoch = 1;

  max_dirty_cache_blocks_ = get_env("PCAS_MAX_DIRTY_CACHE_BLOCKS", 4, global_rank_);

  barrier();
}

template <typename P>
inline pcas_if<P>::~pcas_if() {
  // TODO: calling MPI_Comm_free caused segfault on wisteria-o
  /* MPI_Comm_free(&intra_comm_); */
  /* MPI_Comm_free(&inter_comm_); */

  /* barrier(); */
}

PCAS_TEST_CASE("[pcas::pcas] initialize and finalize PCAS") {
  for (int i = 0; i < 3; i++) {
    pcas pc;
  }
}

template <typename P>
template <typename T>
inline global_ptr<T> pcas_if<P>::malloc(uint64_t    nelems,
                                        dist_policy dpolicy,
                                        uint64_t    block_size [[maybe_unused]]) {
  if (nelems == 0) {
    die("nelems cannot be 0");
  }

  obj_id_t obj_id = obj_id_count_++;

  switch (dpolicy) {
    case dist_policy::block: {
      uint64_t size = nelems * sizeof(T);
      uint64_t local_size = local_block_size(size, global_nproc_);
      uint64_t effective_size = local_size * global_nproc_;

      virtual_mem vm(nullptr, effective_size);
      physical_mem pm_local(local_size, obj_id, intra_rank_, true, true);
      void* vm_local_addr = vm.map_physical_mem(global_rank_ * local_size, 0, local_size, pm_local);

      MPI_Win win = MPI_WIN_NULL;
      MPI_Win_create(vm_local_addr,
                     local_size,
                     1,
                     MPI_INFO_NULL,
                     global_comm_,
                     &win);
      MPI_Win_lock_all(0, win);

      std::vector<physical_mem> pms;
      for (int i = 0; i < intra_nproc_; i++) {
        if (i == intra_rank_) {
          pms.push_back(std::move(pm_local));
        } else {
          int target_rank = intra2global_rank_[i];
          physical_mem pm(local_size, obj_id, i, false, false);
          vm.map_physical_mem(target_rank * local_size, 0, local_size, pm);
          pms.push_back(std::move(pm));
        }
      }

      std::vector<cache_t::entry_t> cache_entries;
      for (uint64_t b = 0; b < effective_size / cache_t::block_size; b++) {
        auto [owner, _idx_b, _idx_e] =
          block_index_info(b * cache_t::block_size, effective_size, global_nproc_);
        if (process_map_[owner].second == inter_rank_) {
          cache_entries.push_back(nullptr);
        } else {
          cache_entries.push_back(cache_.alloc_entry(obj_id));
        }
      }

      obj_entry obe {
        .owner = -1, .id = obj_id,
        .size = size, .effective_size = effective_size,
        .dpolicy = dpolicy, .block_size = local_size,
        .pms = std::move(pms), .vm = std::move(vm),
        .cache_entries = std::move(cache_entries), .win = win,
      };

      auto ret = global_ptr<T>(obe.owner, obe.id, 0);

      objs_[obe.id] = std::move(obe);

      return ret;
    }
    default: {
      die("unimplemented");
      return global_ptr<T>();
    }
  }
}

template <typename P>
template <typename T>
inline void pcas_if<P>::free(global_ptr<T> ptr) {
  if (ptr == global_ptr<T>()) {
    die("null pointer was passed to pcas::free()");
  }
  obj_entry& obe = objs_[ptr.id()];
  switch (obe.dpolicy) {
    case dist_policy::block: {
      for (auto& cae : obe.cache_entries) {
        if (cae) {
          cache_.free_entry(cae);
        }
      }
      MPI_Win_unlock_all(obe.win);
      MPI_Win_free(&obe.win);
      break;
    }
    default: {
      die("unimplemented");
      break;
    }
  }
  objs_.erase(ptr.id());
}

PCAS_TEST_CASE("[pcas::pcas] malloc and free with block policy") {
  pcas pc;
  int n = 10;
  PCAS_SUBCASE("free immediately") {
    for (int i = 1; i < n; i++) {
      auto p = pc.malloc<int>(i * 1234);
      pc.free(p);
    }
  }
  PCAS_SUBCASE("free after accumulation") {
    global_ptr<int> ptrs[n];
    for (int i = 1; i < n; i++) {
      ptrs[i] = pc.malloc<int>(i * 2743);
    }
    for (int i = 1; i < n; i++) {
      pc.free(ptrs[i]);
    }
  }
}

template <typename P>
template <typename T, typename Func>
inline void pcas_if<P>::for_each_block(global_ptr<T> ptr, uint64_t nelems, Func fn) {
  obj_entry& obe = objs_[ptr.id()];

  uint64_t offset_min = ptr.offset();
  uint64_t offset_max = offset_min + nelems * sizeof(T);
  uint64_t offset     = offset_min;

  PCAS_CHECK(offset_max <= obe.size);

  while (offset < offset_max) {
    auto [owner, idx_b, idx_e] = block_index_info(offset, obe.effective_size, global_nproc_);
    uint64_t ib = std::max(idx_b, offset_min);
    uint64_t ie = std::min(idx_e, offset_max);

    fn(owner, ib, ie);

    offset = idx_e;
  }
}

PCAS_TEST_CASE("[pcas::pcas] loop over blocks") {
  pcas pc;

  int nproc = pc.nproc();

  int n = 100000;
  auto p = pc.malloc<int>(n);

  PCAS_SUBCASE("loop over the entire array") {
    int prev_owner = -1;
    uint64_t prev_ie = 0;
    pc.for_each_block(p, n, [&](int owner, uint64_t ib, uint64_t ie) {
      PCAS_CHECK(owner == prev_owner + 1);
      PCAS_CHECK(ib == prev_ie);
      prev_owner = owner;
      prev_ie = ie;
    });
    PCAS_CHECK(prev_owner == nproc - 1);
    PCAS_CHECK(prev_ie == n * sizeof(int));
  }

  PCAS_SUBCASE("loop over the partial array") {
    int b = n / 5 * 2;
    int e = n / 5 * 4;
    int s = e - b;

    auto [o1, _ib1, _ie1] = block_index_info(b * sizeof(int), n * sizeof(int), nproc);
    auto [o2, _ib2, _ie2] = block_index_info(e * sizeof(int), n * sizeof(int), nproc);

    int prev_owner = o1 - 1;
    uint64_t prev_ie = b * sizeof(int);
    pc.for_each_block(p + b, s, [&](int owner, uint64_t ib, uint64_t ie) {
      PCAS_CHECK(owner == prev_owner + 1);
      PCAS_CHECK(ib == prev_ie);
      prev_owner = owner;
      prev_ie = ie;
    });
    auto& o2_ = o2; // structured bindings cannot be captured by lambda until C++20
    PCAS_CHECK(prev_owner == o2_);
    PCAS_CHECK(prev_ie == e * sizeof(int));
  }

  pc.free(p);
}

template <typename P>
template <typename T>
inline void pcas_if<P>::get(global_ptr<T> from_ptr, T* to_ptr, uint64_t nelems) {
  if (from_ptr.owner() == -1) {
    obj_entry& obe = objs_[from_ptr.id()];
    uint64_t offset = from_ptr.offset();
    std::vector<MPI_Request> reqs;

    for_each_block(from_ptr, nelems, [&](int owner, uint64_t idx_b, uint64_t idx_e) {
      MPI_Request req;
      MPI_Rget((uint8_t*)to_ptr - offset + idx_b,
               idx_e - idx_b,
               MPI_UINT8_T,
               owner,
               idx_b - owner * obe.block_size,
               idx_e - idx_b,
               MPI_UINT8_T,
               obe.win,
               &req);
      reqs.push_back(req);
    });

    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
  } else {
    die("unimplemented");
  }
}

template <typename P>
template <typename T>
inline void pcas_if<P>::put(const T* from_ptr, global_ptr<T> to_ptr, uint64_t nelems) {
  if (to_ptr.owner() == -1) {
    obj_entry& obe = objs_[to_ptr.id()];
    uint64_t offset = to_ptr.offset();
    std::vector<MPI_Request> reqs;

    for_each_block(to_ptr, nelems, [&](int owner, uint64_t idx_b, uint64_t idx_e) {
      MPI_Request req;
      MPI_Rput((uint8_t*)from_ptr - offset + idx_b,
               idx_e - idx_b,
               MPI_UINT8_T,
               owner,
               idx_b - owner * obe.block_size,
               idx_e - idx_b,
               MPI_UINT8_T,
               obe.win,
               &req);
      reqs.push_back(req);
    });

    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    // ensure remote completion
    MPI_Win_flush_all(obe.win);
  } else {
    die("unimplemented");
  }
}

PCAS_TEST_CASE("[pcas::pcas] get and put") {
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

  PCAS_SUBCASE("get the entire array") {
    int special = 417;
    buf[0] = buf[n + 1] = special;

    pc.get(p, buf + 1, n);

    for (int i = 0; i < n; i++) {
      PCAS_CHECK(buf[i + 1] == i);
    }
    PCAS_CHECK(buf[0]     == special);
    PCAS_CHECK(buf[n + 1] == special);
  }

  PCAS_SUBCASE("get the partial array") {
    int ib = n / 5 * 2;
    int ie = n / 5 * 4;
    int s = ie - ib;

    int special = 417;
    buf[0] = buf[s + 1] = special;

    pc.get(p + ib, buf + 1, s);

    for (int i = 0; i < s; i++) {
      PCAS_CHECK(buf[i + 1] == i + ib);
    }
    PCAS_CHECK(buf[0]     == special);
    PCAS_CHECK(buf[s + 1] == special);
  }

  PCAS_SUBCASE("get each element") {
    for (int i = 0; i < n; i++) {
      int special = 417;
      buf[0] = buf[2] = special;
      pc.get(p + i, &buf[1], 1);
      PCAS_CHECK(buf[0] == special);
      PCAS_CHECK(buf[1] == i);
      PCAS_CHECK(buf[2] == special);
    }
  }

  pc.free(p);
}

template <typename P>
template <access_mode Mode, typename T>
inline std::conditional_t<Mode == access_mode::read, const T*, T*>
pcas_if<P>::checkout(global_ptr<T> ptr, uint64_t nelems) {
  auto ev = logger::template record<logger_kind::Checkout>(nelems * sizeof(T));

  obj_entry& obe = objs_[ptr.id()];

  // tuple(prev_entry, new_entry, block_num)
  std::vector<std::tuple<cache_t::entry_t, cache_t::entry_t, cache_t::block_num_t>> filled_cache_entries;
  std::vector<MPI_Request> reqs;

  uint64_t cache_entry_b = ptr.offset() / cache_t::block_size;
  uint64_t cache_entry_e =
    (ptr.offset() + nelems * sizeof(T) + cache_t::block_size - 1) / cache_t::block_size;
  for (uint64_t b = cache_entry_b; b < cache_entry_e; b++) {
    auto cae = obe.cache_entries[b];
    if (cae) {
      uint64_t vm_offset = b * cache_t::block_size;
      auto [hit, prev_cae] = cache_.checkout(cae);
      if (!hit) {
        filled_cache_entries.push_back(std::make_tuple(prev_cae, cae, vm_offset));
      }
      if (cae->flushing && Mode != access_mode::read) {
        // MPI_Put has been already started on this cache block.
        // As overlapping MPI_Put calls for the same location will cause undefined behaviour,
        // we need to insert MPI_Win_flush between overlapping MPI_Put calls here.
        auto ev2 = logger::template record<logger_kind::FlushConflicted>();
        complete_flush();
      }
      // Suppose that a cache block is represented as [a1, a2].
      // If a1 is checked out with write-only access mode, then [a1, a2] is allocated a cache entry,
      // but fetch for a1 and a2 is skipped.  Later, if a2 is checked out with read access mode,
      // the data for a2 would not be fetched because it is already in the cache.
      // Thus, we allocate a `fetched` flag to each cache entry to indicate if the entire cache block
      // has been already fetched or not.
      if (Mode != access_mode::write && !cae->fetched) {
        // TODO: fix the assumption cache_t::block_size == min_block_size
        auto [owner, _idx_b, _idx_e] =
          block_index_info(b * cache_t::block_size, obe.effective_size, global_nproc_);

        void* cache_block_ptr = cache_.pm().anon_vm_addr();

        auto& owner_ = owner; // structured bindings cannot be captured by lambda until C++20
        PCAS_CHECK(vm_offset >= owner_ * obe.block_size);
        PCAS_CHECK(vm_offset - owner_ * obe.block_size + cache_t::block_size <= obe.block_size);

        // fetch only nondirty sections
        for (auto [offset_in_block_b, offset_in_block_e] :
             sections_inverse(cae->dirty_sections, {0, cache_t::block_size})) {
          MPI_Request req;
          MPI_Rget((uint8_t*)cache_block_ptr + cae->block_num * cache_t::block_size + offset_in_block_b,
                   offset_in_block_e - offset_in_block_b,
                   MPI_UINT8_T,
                   owner,
                   vm_offset - owner * obe.block_size + offset_in_block_b,
                   offset_in_block_e - offset_in_block_b,
                   MPI_UINT8_T,
                   obe.win,
                   &req);
          reqs.push_back(req);
        }

        cae->fetched = true;
      }
    }
  }

  // Overlap communication and memory remapping
  for (auto [prev_cae, new_cae, vm_offset] : filled_cache_entries) {
    if (prev_cae) {
      PCAS_CHECK(prev_cae->vm_addr);
      virtual_mem::mmap_no_physical_mem(prev_cae->vm_addr, cache_t::block_size);
    }
    physical_mem& cache_pm = cache_.pm();
    obe.vm.map_physical_mem(vm_offset, new_cae->block_num * cache_t::block_size, cache_t::block_size, cache_pm);
    new_cae->vm_addr = (uint8_t*)obe.vm.addr() + vm_offset;
  }

  MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

  T* ret = (T*)((uint8_t*)obe.vm.addr() + ptr.offset());

  checkouts_[(void*)ret] = (checkout_entry){
    .ptr = static_cast<global_ptr<uint8_t>>(ptr), .raw_ptr = (uint8_t*)ret,
    .size = nelems * sizeof(T), .mode = Mode,
  };

  return ret;
}

template <typename P>
template <typename T>
inline void pcas_if<P>::checkin(T* raw_ptr) {
  auto ev = logger::template record<logger_kind::Checkin>();

  auto c = checkouts_.find((void*)raw_ptr);
  if (c == checkouts_.end()) {
    die("The pointer %p passed to checkin() is not registered", raw_ptr);
  }

  checkout_entry che = c->second;
  obj_entry& obe = objs_[che.ptr.id()];

  uint64_t cache_entry_b = che.ptr.offset() / cache_t::block_size;
  uint64_t cache_entry_e =
    (che.ptr.offset() + che.size + cache_t::block_size - 1) / cache_t::block_size;

  if (che.mode == access_mode::read_write ||
      che.mode == access_mode::write) {
    for (uint64_t b = cache_entry_b; b < cache_entry_e; b++) {
      auto cae = obe.cache_entries[b];
      if (cae) {
        n_dirty_cache_blocks_ += cae->dirty_sections.empty();
        uint64_t offset_in_block_b = (che.ptr.offset() > b * cache_t::block_size) ?
                                     che.ptr.offset() - b * cache_t::block_size : 0;
        uint64_t offset_in_block_e = (che.ptr.offset() + che.size < (b + 1) * cache_t::block_size) ?
                                     che.ptr.offset() + che.size - b * cache_t::block_size : cache_t::block_size;
        sections_insert(cae->dirty_sections, {offset_in_block_b, offset_in_block_e});
        cache_dirty_ = true;
      }
    }
  }

  for (uint64_t b = cache_entry_b; b < cache_entry_e; b++) {
    auto cae = obe.cache_entries[b];
    if (cae) {
      cache_.checkin(cae);
      // If the entire cache block is written, we consider that it is already fetched.
      // If only a part of the block is written, we need to fetch the block when this block
      // is checked out with read access mode.
      if (che.mode == access_mode::write &&
          che.ptr.offset() <= b * cache_t::block_size &&
          (b + 1) * cache_t::block_size <= che.ptr.offset() + che.size) {
        cae->fetched = true;
      }
    }
  }

  checkouts_.erase((void*)raw_ptr);
}

template <typename P>
template <typename T>
inline void pcas_if<P>::checkin(const T* raw_ptr) {
  checkin(const_cast<T*>(raw_ptr));
}

PCAS_TEST_CASE("[pcas::pcas] checkout and checkin (small, aligned)") {
  pcas pc;

  int rank = pc.rank();
  int nproc = pc.nproc();

  int n = min_block_size * nproc;
  auto p = pc.malloc<uint8_t>(n);

  uint8_t* home_ptr = (uint8_t*)pc.get_physical_mem(p);
  for (uint64_t i = 0; i < min_block_size; i++) {
    home_ptr[i] = rank;
  }

  pc.barrier();

  PCAS_SUBCASE("read the entire array") {
    const uint8_t* rp = pc.checkout<access_mode::read>(p, n);
    for (int i = 0; i < n; i++) {
      PCAS_CHECK_MESSAGE(rp[i] == i / min_block_size, "rank: ", rank, ", i: ", i);
    }
    pc.checkin(rp);
  }

  PCAS_SUBCASE("read and write the entire array") {
    for (int it = 0; it < nproc; it++) {
      if (it == rank) {
        uint8_t* rp = pc.checkout<access_mode::read_write>(p, n);
        for (int i = 0; i < n; i++) {
          PCAS_CHECK_MESSAGE(rp[i] == i / min_block_size + it, "it: ", it, ", rank: ", rank, ", i: ", i);
          rp[i]++;
        }
        pc.checkin(rp);
      }
      pc.barrier();

      const uint8_t* rp = pc.checkout<access_mode::read>(p, n);
      for (int i = 0; i < n; i++) {
        PCAS_CHECK_MESSAGE(rp[i] == i / min_block_size + it + 1, "it: ", it, ", rank: ", rank, ", i: ", i);
      }
      pc.checkin(rp);

      pc.barrier();
    }
  }

  PCAS_SUBCASE("read the partial array") {
    int ib = n / 5 * 2;
    int ie = n / 5 * 4;
    int s = ie - ib;

    const uint8_t* rp = pc.checkout<access_mode::read>(p + ib, s);
    for (int i = 0; i < s; i++) {
      PCAS_CHECK_MESSAGE(rp[i] == (i + ib) / min_block_size, "rank: ", rank, ", i: ", i);
    }
    pc.checkin(rp);
  }

  pc.free(p);
}

PCAS_TEST_CASE("[pcas::pcas] checkout and checkin (large, not aligned)") {
  pcas pc(16 * min_block_size);

  int rank = pc.rank();
  int nproc = pc.nproc();

  int n = 100000;
  auto p = pc.malloc<int>(n);

  int max_checkout_size = (16 - 2) * min_block_size / sizeof(int);

  if (rank == 0) {
    for (int i = 0; i < n; i += max_checkout_size) {
      int m = std::min(max_checkout_size, n - i);
      int* rp = pc.checkout<access_mode::write>(p + i, m);
      for (int j = 0; j < m; j++) {
        rp[j] = i + j;
      }
      pc.checkin(rp);
    }
  }

  pc.barrier();

  PCAS_SUBCASE("read the entire array") {
    for (int i = 0; i < n; i += max_checkout_size) {
      int m = std::min(max_checkout_size, n - i);
      const int* rp = pc.checkout<access_mode::read>(p + i, m);
      for (int j = 0; j < m; j++) {
        PCAS_CHECK(rp[j] == i + j);
      }
      pc.checkin(rp);
    }
  }

  PCAS_SUBCASE("read the partial array") {
    int ib = n / 5 * 2;
    int ie = n / 5 * 4;
    int s = ie - ib;

    for (int i = 0; i < s; i += max_checkout_size) {
      int m = std::min(max_checkout_size, s - i);
      const int* rp = pc.checkout<access_mode::read>(p + ib + i, m);
      for (int j = 0; j < m; j++) {
        PCAS_CHECK(rp[j] == i + ib + j);
      }
      pc.checkin(rp);
    }
  }

  PCAS_SUBCASE("read and write the partial array") {
    int stride = 48;
    PCAS_REQUIRE(stride <= max_checkout_size);
    for (int i = rank * stride; i < n; i += nproc * stride) {
      int s = std::min(stride, n - i);
      int* rp = pc.checkout<access_mode::read_write>(p + i, s);
      for (int j = 0; j < s; j++) {
        PCAS_CHECK(rp[j] == j + i);
        rp[j] *= 2;
      }
      pc.checkin(rp);
    }

    pc.barrier();

    for (int i = 0; i < n; i += max_checkout_size) {
      int m = std::min(max_checkout_size, n - i);
      const int* rp = pc.checkout<access_mode::read>(p + i, m);
      for (int j = 0; j < m; j++) {
        PCAS_CHECK(rp[j] == (i + j) * 2);
      }
      pc.checkin(rp);
    }
  }

  pc.free(p);
}

}
