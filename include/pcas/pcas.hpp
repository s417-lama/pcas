#pragma once

#include <type_traits>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <functional>

#include <mpi.h>

#include "pcas/util.hpp"
#include "pcas/global_ptr.hpp"
#include "pcas/physical_mem.hpp"
#include "pcas/virtual_mem.hpp"
#include "pcas/cache.hpp"
#include "pcas/wallclock.hpp"
#include "pcas/logger/logger.hpp"
#include "pcas/mem_mapper.hpp"

namespace pcas {

enum class access_mode {
  read,
  write,
  read_write,
};

using epoch_t = uint64_t;

struct release_handler {
  int rank;
  epoch_t epoch;
};

struct policy_default {
  using wallclock_t = wallclock_native;
  using logger_kind_t = logger::kind;
  template <typename P>
  using logger_impl_t = logger::impl_dummy<P>;
  constexpr static uint64_t block_size = 65536;
};

template <typename P>
class pcas_if;

using pcas = pcas_if<policy_default>;

template <typename P>
class pcas_if {
  using this_t = pcas_if<P>;

  using block_num_t = uint64_t;

  enum class cache_state {
    unmapped,   // initial state
    invalid,    // this entry is in the cache, but the data is not up-to-date
    fetching,   // communication (read) is in-progress
    valid,      // the data is up-to-date
  };

  struct home_block {
    cache_entry_num_t entry_num = std::numeric_limits<cache_entry_num_t>::max();
    physical_mem*     pm;
    uint64_t          pm_offset;
    uint8_t*          vm_addr;
    uint64_t          size;
    int               checkout_count = 0;
    uint8_t*          prev_vm_addr   = nullptr;
    bool              mapped         = false;

    void map() {
      pm->map(vm_addr, pm_offset, size);
      mapped = true;
    }

    void unmap_prev() {
      if (prev_vm_addr) {
        virtual_mem::mmap_no_physical_mem(prev_vm_addr, size);
        prev_vm_addr = nullptr;
      }
    }

    /* Callback functions for cache_system class */

    bool is_evictable() const {
      return checkout_count == 0;
    }

    void on_evict() {
      PCAS_CHECK(is_evictable());
      PCAS_CHECK(prev_vm_addr == nullptr);
      prev_vm_addr = (uint8_t*)vm_addr;
      mapped = false;
    }

    void on_cache_map(cache_entry_num_t b) {
      entry_num = b;
    }
  };

  using mmap_cache_t = cache_system<uintptr_t, home_block>;

  struct cache_block {
    cache_entry_num_t entry_num      = std::numeric_limits<cache_entry_num_t>::max();
    cache_state       cstate         = cache_state::unmapped;
    bool              transitive     = false;
    bool              flushing       = false;
    int               checkout_count = 0;
    uint8_t*          vm_addr        = nullptr;
    uint8_t*          prev_vm_addr   = nullptr;
    bool              mapped         = false;
    MPI_Request       req            = MPI_REQUEST_NULL;
    obj_id_t          obj_id;
    sections          dirty_sections;
    sections          partial_sections; // for write-only update

    void invalidate() {
      if (cstate == cache_state::fetching) {
        // Only for prefetching blocks
        PCAS_CHECK(req != MPI_REQUEST_NULL);
        // FIXME: MPI_Cancel causes segfault
        /* MPI_Cancel(&req); */
        /* MPI_Request_free(&req); */
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        PCAS_CHECK(req == MPI_REQUEST_NULL);
      }
      PCAS_CHECK(dirty_sections.empty());
      partial_sections.clear();
      cstate = cache_state::invalid;
      PCAS_CHECK(is_evictable());
    }

    void map(physical_mem& cache_pm) {
      cache_pm.map(vm_addr, entry_num * block_size, block_size);
      mapped = true;
    }

    void unmap_prev() {
      if (prev_vm_addr) {
        virtual_mem::mmap_no_physical_mem(prev_vm_addr, block_size);
        prev_vm_addr = nullptr;
      }
    }

    /* Callback functions for cache_system class */

    bool is_evictable() const {
      return checkout_count == 0 && !transitive &&
             !flushing && dirty_sections.empty();
    }

    void on_evict() {
      PCAS_CHECK(is_evictable());
      invalidate();
      entry_num = std::numeric_limits<cache_entry_num_t>::max();
      cstate = cache_state::unmapped;
      PCAS_CHECK(prev_vm_addr == nullptr);
      prev_vm_addr = (uint8_t*)vm_addr;
      mapped = false;
    }

    void on_cache_map(cache_entry_num_t b) {
      entry_num = b;
      cstate = cache_state::invalid;
    }
  };

  using cache_t = cache_system<uintptr_t, cache_block>;

  struct mem_obj {
    int                                   owner;
    obj_id_t                              id;
    uint64_t                              size;
    uint64_t                              effective_size;
    std::unique_ptr<mem_mapper::base>     mmapper;
    std::unordered_map<int, physical_mem> home_pms;
    virtual_mem                           vm;
    block_num_t                           last_checkout_block_num;
    MPI_Win                               win;
  };

  struct checkout_entry {
    global_ptr<uint8_t> ptr;
    access_mode         mode;
    uint64_t            count;
  };

  struct release_remote_region {
    epoch_t request; // requested epoch from remote
    epoch_t epoch; // current epoch of the owner
  };

  struct release_manager {
    release_remote_region* remote;
    MPI_Win win;

    release_manager(MPI_Comm comm) {
      MPI_Win_allocate(sizeof(release_remote_region),
                       1,
                       MPI_INFO_NULL,
                       comm,
                       &remote,
                       &win);
      MPI_Win_lock_all(0, win);

      remote->request = 1;
      remote->epoch = 1;
    }

    ~release_manager() {
      MPI_Win_unlock_all(win);
      MPI_Win_free(&win);
    }
  };

  struct comm_group {
    int      rank  = -1;
    int      nproc = -1;
    MPI_Comm comm  = MPI_COMM_NULL;

    comm_group(MPI_Comm c) : comm(c) {
      MPI_Comm_rank(c, &rank);
      MPI_Comm_size(c, &nproc);
      PCAS_CHECK(rank != -1);
      PCAS_CHECK(nproc != -1);
    }
  };

  // Member variables
  // -----------------------------------------------------------------------------

  bool validate_dummy_; // for initialization

  comm_group cg_global_;
  comm_group cg_intra_;
  comm_group cg_inter_;

  std::vector<std::pair<int, int>> process_map_; // pair: (intra, inter rank)
  std::vector<int> intra2global_rank_;

  obj_id_t obj_id_count_ = 1; // TODO: better management of used IDs
  std::unordered_map<obj_id_t, mem_obj> objs_;

  // FIXME: using map instead of unordered_map because std::pair needs a user-defined hash function
  std::map<std::pair<void*, uint64_t>, checkout_entry> checkouts_;

  mmap_cache_t mmap_cache_;
  cache_t cache_;
  physical_mem cache_pm_;

  std::vector<home_block*> remap_home_blocks_;
  std::vector<cache_block*> remap_cache_blocks_;

  bool cache_dirty_ = false;
  release_manager rm_;

  std::unordered_set<MPI_Win> flushing_wins_;

  uint64_t n_dirty_cache_blocks_ = 0;
  uint64_t max_dirty_cache_blocks_;

  int enable_shared_memory_;
  int enable_write_through_;
  int n_prefetch_;

  // Initializaiton
  // -----------------------------------------------------------------------------

  bool validate_input(uint64_t cache_size, MPI_Comm comm) {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
      die("Please call MPI_Init() before initializing PCAS.");
    }

    uint64_t pagesize = sysconf(_SC_PAGE_SIZE);
    if (block_size == 0 || block_size % pagesize != 0) {
      die("The block size (%ld) must be a multiple of the system page size (%ld).", block_size, pagesize);
    }

    if (cache_size % block_size != 0) {
      die("The cache size (%ld) must be a multiple of the block size (%ld).", cache_size, block_size);
    }

    if (comm == MPI_COMM_NULL) {
      die("MPI_COMM_NULL is given.");
    }

    return true;
  }

  MPI_Comm create_intra_comm() {
    MPI_Comm h;
    MPI_Comm_split_type(cg_global_.comm, MPI_COMM_TYPE_SHARED, cg_global_.rank, MPI_INFO_NULL, &h);
    return h;
  }

  MPI_Comm create_inter_comm() {
    MPI_Comm h;
    MPI_Comm_split(cg_global_.comm, cg_intra_.rank, cg_global_.rank, &h);
    return h;
  }

  std::vector<std::pair<int, int>> create_process_map() {
    std::pair<int, int> myranks {cg_intra_.rank, cg_inter_.rank};
    std::vector<std::pair<int, int>> ret(cg_global_.nproc);
    MPI_Allgather(&myranks,
                  sizeof(std::pair<int, int>),
                  MPI_BYTE,
                  ret.data(),
                  sizeof(std::pair<int, int>),
                  MPI_BYTE,
                  cg_global_.comm);
    return ret;
  }

  std::vector<int> create_intra2global_rank() {
    std::vector<int> ret;
    for (int i = 0; i < cg_global_.nproc; i++) {
      if (process_map_[i].second == cg_inter_.rank) {
        ret.push_back(i);
      }
    }
    PCAS_CHECK(ret.size() == (size_t)cg_intra_.nproc);
    return ret;
  }

  uint64_t get_home_mmap_limit(uint64_t n_cache_blocks) {
    uint64_t sys_limit = sys_mmap_entry_limit();
    uint64_t margin = 1000;
    PCAS_CHECK(sys_limit > n_cache_blocks + margin);
    return (sys_limit - n_cache_blocks - margin) / 2;
  }

  bool is_locally_accessible(int owner) {
    return owner == cg_global_.rank ||
           (enable_shared_memory_ && process_map_[owner].second == cg_inter_.rank);
  }

  void ensure_remapped() {
    if (!remap_home_blocks_.empty()) {
      for (home_block* hb : remap_home_blocks_) {
        PCAS_CHECK(hb->entry_num < mmap_cache_.num_entries());
        // save the number of mmap entries by unmapping previous virtual memory
        hb->unmap_prev();
        hb->map();
      }
      remap_home_blocks_.clear();
    }
    if (!remap_cache_blocks_.empty()) {
      for (cache_block* cb : remap_cache_blocks_) {
        PCAS_CHECK(cb->entry_num < cache_.num_entries());
        // save the number of mmap entries by unmapping previous virtual memory
        cb->unmap_prev();
        cb->map(cache_pm_);
      }
      remap_cache_blocks_.clear();
    }
  }

  void fetch_begin(mem_obj& mo, cache_block& cb, int owner, size_t owner_disp) {
    void* cache_block_ptr = cache_pm_.anon_vm_addr();
    int count = 0;
    // fetch only nondirty sections
    for (auto [offset_in_block_b, offset_in_block_e] :
         sections_inverse(cb.partial_sections, {0, block_size})) {
      PCAS_CHECK(cb.entry_num < cache_.num_entries());
      MPI_Request req;
      MPI_Rget((uint8_t*)cache_block_ptr + cb.entry_num * block_size + offset_in_block_b,
               offset_in_block_e - offset_in_block_b,
               MPI_UINT8_T,
               owner,
               owner_disp + offset_in_block_b,
               offset_in_block_e - offset_in_block_b,
               MPI_UINT8_T,
               mo.win,
               &req);
      cb.cstate = cache_state::fetching;
      // FIXME
      if (count == 0) {
        PCAS_CHECK(cb.req == MPI_REQUEST_NULL);
        cb.req = req;
      } else {
        MPI_Wait(&req, MPI_STATUS_IGNORE);
      }
      count++;
    }
  }

  void flush_dirty_cache() {
    if (cache_dirty_) {
      cache_.for_each_entry([&](cache_block& cb) {
        if (cb.mapped && !cb.dirty_sections.empty()) {
          PCAS_CHECK(!cb.flushing);

          mem_obj& mo = objs_[cb.obj_id];
          void* cache_block_ptr = cache_pm_.anon_vm_addr();
          uint64_t offset = cb.vm_addr - (uint8_t*)mo.vm.addr();
          auto bi = mo.mmapper->get_block_info(offset);
          uint64_t pm_offset = bi.pm_offset + offset - bi.offset_b;

          for (auto [offset_in_block_b, offset_in_block_e] : cb.dirty_sections) {
            MPI_Put((uint8_t*)cache_block_ptr + cb.entry_num * block_size + offset_in_block_b,
                    offset_in_block_e - offset_in_block_b,
                    MPI_UINT8_T,
                    bi.owner,
                    pm_offset + offset_in_block_b,
                    offset_in_block_e - offset_in_block_b,
                    MPI_UINT8_T,
                    mo.win);
          }
          cb.dirty_sections.clear();
          cb.flushing = true;

          flushing_wins_.insert(mo.win);
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

      cache_.for_each_entry([&](cache_block& cb) {
        cb.flushing = false;
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

  void cache_invalidate_all() {
    cache_.for_each_entry([&](cache_block& cb) {
      cb.invalidate();
    });
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

  void send_release_request(int target_rank, epoch_t remote_epoch, epoch_t request_epoch) {
    do {
      epoch_t prev;
      MPI_Compare_and_swap(&request_epoch,
                           &remote_epoch,
                           &prev,
                           MPI_UINT64_T, // should match epoch_t
                           target_rank,
                           offsetof(release_remote_region, request),
                           rm_.win);
      MPI_Win_flush(target_rank, rm_.win);
      remote_epoch = prev;
    } while (remote_epoch < request_epoch);
    // FIXME: MPI_Fetch_and_op + MPI_MAX seems not offloaded to RDMA NICs,
    // which consumes too much resources. So currently MPI_Compare_and_swap is used instead.
    // Fix to use fetch_and_op once MPI_MAX gets offloaded to RDMA NICs.
    /* MPI_Fetch_and_op(&request, */
    /*                  &prev, */
    /*                  MPI_UINT64_T, // should match epoch_t */
    /*                  target_rank, */
    /*                  offsetof(release_remote_region, request), */
    /*                  MPI_MAX, */
    /*                  rm_.win); */
    /* MPI_Win_flush(target_rank, rm_.win); */
  }

public:
  using logger = typename logger::template logger_if<logger::policy<P>>;
  using logger_kind = typename P::logger_kind_t::value;

  constexpr static uint64_t block_size = P::block_size;

  pcas_if(uint64_t cache_size = 1024 * block_size, MPI_Comm comm = MPI_COMM_WORLD);
  ~pcas_if();

  int rank() const { return cg_global_.rank; }
  int nproc() const { return cg_global_.nproc; }

  template <typename T, template<uint64_t> typename MemMapper = mem_mapper::cyclic, typename... MemMapperArgs>
  global_ptr<T> malloc(uint64_t nelems, MemMapperArgs... mmargs);

  template <typename T>
  void free(global_ptr<T> ptr);

  template <typename T, typename Func>
  void for_each_block(global_ptr<T> ptr, uint64_t nelems, Func fn);

  template <typename T>
  void get(global_ptr<T> from_ptr, T* to_ptr, uint64_t nelems);

  template <typename T>
  void put(const T* from_ptr, global_ptr<T> to_ptr, uint64_t nelems);

  template <typename T>
  void willread(global_ptr<T> ptr, uint64_t nelems);

  template <access_mode Mode, typename T>
  std::conditional_t<Mode == access_mode::read, const T*, T*>
  checkout(global_ptr<T> ptr, uint64_t nelems);

  template <typename T>
  void checkin(T* raw_ptr, uint64_t nelems);

  void release();

  void release_lazy(release_handler* handler);

  void acquire(release_handler handler = {.rank = 0, .epoch = 0});

  void barrier();

  void poll();

  /* unsafe APIs for debugging */

  template <typename T>
  void* get_physical_mem(global_ptr<T> ptr) {
    mem_obj& mo = objs_[ptr.id()];
    return mo.home_pms[cg_global_.rank].anon_vm_addr();
  }

};

template <typename P>
inline pcas_if<P>::pcas_if(uint64_t cache_size, MPI_Comm comm)
  : validate_dummy_(validate_input(cache_size, comm)),
    cg_global_(comm),
    cg_intra_(create_intra_comm()),
    cg_inter_(create_inter_comm()),
    process_map_(create_process_map()),
    intra2global_rank_(create_intra2global_rank()),
    mmap_cache_(get_home_mmap_limit(cache_size / block_size)),
    cache_(cache_size / block_size),
    cache_pm_(cache_size, 0, cg_intra_.rank, true, true),
    rm_(comm),
    max_dirty_cache_blocks_(get_env("PCAS_MAX_DIRTY_CACHE_SIZE", cache_size / 4, cg_global_.rank) / block_size),
    enable_shared_memory_(get_env("PCAS_ENABLE_SHARED_MEMORY", 1, cg_global_.rank)),
    enable_write_through_(get_env("PCAS_ENABLE_WRITE_THROUGH", 0, cg_global_.rank)),
    n_prefetch_(get_env("PCAS_PREFETCH_BLOCKS", 0, cg_global_.rank)) {

  logger::init(cg_global_.rank, cg_global_.nproc);

  barrier();
}

template <typename P>
inline pcas_if<P>::~pcas_if() {
  MPI_Comm_free(&cg_intra_.comm);
  MPI_Comm_free(&cg_inter_.comm);
  /* barrier(); */
}

PCAS_TEST_CASE("[pcas::pcas] initialize and finalize PCAS") {
  for (int i = 0; i < 3; i++) {
    pcas pc;
  }
}

template <typename P>
template <typename T, template<uint64_t> typename MemMapper, typename... MemMapperArgs>
inline global_ptr<T> pcas_if<P>::malloc(uint64_t nelems, MemMapperArgs... mmargs) {
  if (nelems == 0) {
    die("nelems cannot be 0");
  }

  uint64_t size = nelems * sizeof(T);

  std::unique_ptr<MemMapper<block_size>> mmapper(
    new MemMapper<block_size>(size, cg_global_.nproc, mmargs...));

  uint64_t local_size = mmapper->get_local_size(cg_global_.rank);
  uint64_t effective_size = mmapper->get_effective_size();

  obj_id_t obj_id = obj_id_count_++;

  virtual_mem vm(nullptr, effective_size);
  physical_mem pm_local(local_size, obj_id, cg_intra_.rank, true, true);

  MPI_Win win = MPI_WIN_NULL;
  MPI_Win_create(pm_local.anon_vm_addr(),
                 local_size,
                 1,
                 MPI_INFO_NULL,
                 cg_global_.comm,
                 &win);
  MPI_Win_lock_all(0, win);

  // Open home physical memory of other intra-node processes
  std::unordered_map<int, physical_mem> home_pms;
  for (int i = 0; i < cg_intra_.nproc; i++) {
    if (i == cg_intra_.rank) {
      home_pms[cg_global_.rank] = std::move(pm_local);
    } else if (enable_shared_memory_) {
      int target_rank = intra2global_rank_[i];
      int target_local_size = mmapper->get_local_size(target_rank);
      physical_mem pm(target_local_size, obj_id, i, false, false);
      home_pms[target_rank] = std::move(pm);
    }
  }

  mem_obj mo {
    .owner = -1, .id = obj_id,
    .size = size, .effective_size = effective_size,
    .mmapper = std::move(mmapper),
    .home_pms = std::move(home_pms), .vm = std::move(vm),
    .last_checkout_block_num = std::numeric_limits<block_num_t>::max(),
    .win = win,
  };

  auto ret = global_ptr<T>(mo.owner, mo.id, 0);

  objs_[mo.id] = std::move(mo);

  return ret;
}

template <typename P>
template <typename T>
inline void pcas_if<P>::free(global_ptr<T> ptr) {
  if (ptr == global_ptr<T>()) {
    die("null pointer was passed to pcas::free()");
  }

  // ensure free safety
  ensure_remapped();
  complete_flush();

  mem_obj& mo = objs_[ptr.id()];

  // ensure all cache entries are evicted
  for (uint64_t offset = 0; offset < mo.effective_size; offset += block_size) {
    void* vm_addr = (uint8_t*)mo.vm.addr() + offset;
    mmap_cache_.ensure_evicted((uintptr_t)vm_addr / block_size);
    cache_.ensure_evicted((uintptr_t)vm_addr / block_size);
  }

  MPI_Win_unlock_all(mo.win);
  MPI_Win_free(&mo.win);
  objs_.erase(ptr.id());
}

PCAS_TEST_CASE("[pcas::pcas] malloc and free with block policy") {
  pcas pc;
  int n = 10;
  PCAS_SUBCASE("free immediately") {
    for (int i = 1; i < n; i++) {
      auto p = pc.malloc<int, mem_mapper::block>(i * 1234);
      pc.free(p);
    }
  }
  PCAS_SUBCASE("free after accumulation") {
    global_ptr<int> ptrs[n];
    for (int i = 1; i < n; i++) {
      ptrs[i] = pc.malloc<int, mem_mapper::block>(i * 2743);
    }
    for (int i = 1; i < n; i++) {
      pc.free(ptrs[i]);
    }
  }
}

PCAS_TEST_CASE("[pcas::pcas] malloc and free with cyclic policy") {
  pcas pc;
  int n = 10;
  PCAS_SUBCASE("free immediately") {
    for (int i = 1; i < n; i++) {
      auto p = pc.malloc<int, mem_mapper::cyclic>(i * 123456);
      pc.free(p);
    }
  }
  PCAS_SUBCASE("free after accumulation") {
    global_ptr<int> ptrs[n];
    for (int i = 1; i < n; i++) {
      ptrs[i] = pc.malloc<int, mem_mapper::cyclic>(i * 27438, pcas::block_size * i);
    }
    for (int i = 1; i < n; i++) {
      pc.free(ptrs[i]);
    }
  }
}

template <typename P>
template <typename T, typename Func>
inline void pcas_if<P>::for_each_block(global_ptr<T> ptr, uint64_t nelems, Func fn) {
  mem_obj& mo = objs_[ptr.id()];

  uint64_t offset_min = ptr.offset();
  uint64_t offset_max = offset_min + nelems * sizeof(T);
  uint64_t offset     = offset_min;

  PCAS_CHECK(offset_max <= mo.size);

  while (offset < offset_max) {
    auto bi = mo.mmapper->get_block_info(offset);
    fn(bi);
    offset = bi.offset_e;
  }
}

PCAS_TEST_CASE("[pcas::pcas] loop over blocks") {
  pcas pc;

  int nproc = pc.nproc();

  int n = 1000000;
  auto p = pc.malloc<int, mem_mapper::block>(n);

  PCAS_SUBCASE("loop over the entire array") {
    int prev_owner = -1;
    uint64_t prev_offset_e = 0;
    pc.for_each_block(p, n, [&](const auto& bi) {
      PCAS_CHECK(bi.owner == prev_owner + 1);
      PCAS_CHECK(bi.offset_b == prev_offset_e);
      prev_owner = bi.owner;
      prev_offset_e = bi.offset_e;
    });
    PCAS_CHECK(prev_owner == nproc - 1);
    PCAS_CHECK(prev_offset_e >= n * sizeof(int));
  }

  PCAS_SUBCASE("loop over the partial array") {
    int b = n / 5 * 2;
    int e = n / 5 * 4;
    int s = e - b;

    mem_mapper::block<pcas::block_size> mmapper{n * sizeof(int), nproc};
    auto bi1 = mmapper.get_block_info(b * sizeof(int));
    auto bi2 = mmapper.get_block_info(e * sizeof(int));

    int prev_owner = bi1.owner - 1;
    uint64_t prev_offset_e = bi1.offset_b;
    pc.for_each_block(p + b, s, [&](const auto& bi) {
      PCAS_CHECK(bi.owner == prev_owner + 1);
      PCAS_CHECK(bi.offset_b == prev_offset_e);
      prev_owner = bi.owner;
      prev_offset_e = bi.offset_e;
    });
    PCAS_CHECK(prev_owner == bi2.owner);
    PCAS_CHECK(prev_offset_e == bi2.offset_e);
  }

  pc.free(p);
}

template <typename P>
template <typename T>
inline void pcas_if<P>::get(global_ptr<T> from_ptr, T* to_ptr, uint64_t nelems) {
  if (from_ptr.owner() == -1) {
    mem_obj& mo = objs_[from_ptr.id()];
    std::vector<MPI_Request> reqs;

    uint64_t offset_min = from_ptr.offset();
    uint64_t offset_max = offset_min + nelems * sizeof(T);

    for_each_block(from_ptr, nelems, [&](const auto& bi) {
      uint64_t offset_b = std::max(bi.offset_b, offset_min);
      uint64_t offset_e = std::min(bi.offset_e, offset_max);
      uint64_t pm_offset = bi.pm_offset + offset_b - bi.offset_b;
      MPI_Request req;
      MPI_Rget((uint8_t*)to_ptr - offset_min + offset_b,
               offset_e - offset_b,
               MPI_UINT8_T,
               bi.owner,
               pm_offset,
               offset_e - offset_b,
               MPI_UINT8_T,
               mo.win,
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
    mem_obj& mo = objs_[to_ptr.id()];

    uint64_t offset_min = to_ptr.offset();
    uint64_t offset_max = offset_min + nelems * sizeof(T);

    for_each_block(to_ptr, nelems, [&](const auto& bi) {
      uint64_t offset_b = std::max(bi.offset_b, offset_min);
      uint64_t offset_e = std::min(bi.offset_e, offset_max);
      uint64_t pm_offset = bi.pm_offset + offset_b - bi.offset_b;
      MPI_Put((uint8_t*)from_ptr - offset_min + offset_b,
              offset_e - offset_b,
              MPI_UINT8_T,
              bi.owner,
              pm_offset,
              offset_e - offset_b,
              MPI_UINT8_T,
              mo.win);
    });

    // ensure remote completion
    MPI_Win_flush_all(mo.win);
  } else {
    die("unimplemented");
  }
}

PCAS_TEST_CASE("[pcas::pcas] get and put") {
  pcas pc;

  int rank = pc.rank();

  int n = 1000000;

  global_ptr<int> ps[2];
  ps[0] = pc.malloc<int, mem_mapper::block >(n);
  ps[1] = pc.malloc<int, mem_mapper::cyclic>(n);

  int* buf = new int[n + 2];

  for (auto p : ps) {
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
  }

  delete[] buf;

  pc.free(ps[0]);
  pc.free(ps[1]);
}

template <typename P>
template <typename T>
inline void pcas_if<P>::willread(global_ptr<T> ptr, uint64_t nelems) {
  uint64_t size = nelems * sizeof(T);
  auto ev = logger::template record<logger_kind::Willread>(size);

  mem_obj& mo = objs_[ptr.id()];

  section block_section{0, block_size};

  uint64_t offset_max = ptr.offset() + size;

  // get cache blocks first, otherwise the same cache block may be assigned to multiple mem blocks
  for_each_block(ptr, nelems, [&](const auto& bi) {
    if (!is_locally_accessible(bi.owner)) {
      uint64_t offset_b = std::max(bi.offset_b, ptr.offset() / block_size * block_size);
      uint64_t offset_e = std::min({bi.offset_e, ptr.offset() + nelems * sizeof(T), offset_max});
      for (uint64_t offset = offset_b; offset < offset_e; offset += block_size) {
        uint8_t* vm_addr = (uint8_t*)mo.vm.addr() + offset;

        try {
          cache_block& cb = cache_.ensure_cached(vm_addr);

          if (!cb.mapped) {
            cb.vm_addr = vm_addr;
            cb.obj_id = mo.id;
            remap_cache_blocks_.push_back(&cb);
          }

          cb.transitive = true;
        } catch (cache_full_exception& e) {
          offset_max = offset;
          break;
        }
      }
    }
  });

  for_each_block(ptr, nelems, [&](const auto& bi) {
    if (!is_locally_accessible(bi.owner)) {
      uint64_t offset_b = std::max(bi.offset_b, ptr.offset() / block_size * block_size);
      uint64_t offset_e = std::min({bi.offset_e, ptr.offset() + size, offset_max});
      for (uint64_t offset = offset_b; offset < offset_e; offset += block_size) {
        uint8_t* vm_addr = (uint8_t*)mo.vm.addr() + offset;
        cache_block& cb = cache_.ensure_cached(vm_addr);

        fetch_begin(mo, cb, bi.owner, bi.pm_offset + offset - bi.offset_b);
        sections_insert(cb.partial_sections, block_section);

        PCAS_CHECK(cb.transitive);
        cb.transitive = false;
      }
    }
  });
}

template <typename P>
template <access_mode Mode, typename T>
inline std::conditional_t<Mode == access_mode::read, const T*, T*>
pcas_if<P>::checkout(global_ptr<T> ptr, uint64_t nelems) {
  uint64_t size = nelems * sizeof(T);
  auto ev = logger::template record<logger_kind::Checkout>(size);

  mem_obj& mo = objs_[ptr.id()];

  int n_prefetch = 0;
  if (Mode != access_mode::write) {
    block_num_t block_num_b = ptr.offset() / block_size;
    block_num_t block_num_e = (ptr.offset() + size + block_size - 1) / block_size;
    if (block_num_b <= mo.last_checkout_block_num + 1 &&
        mo.last_checkout_block_num + 1 < block_num_e) {
      // If it seems sequential access, do prefetch
      n_prefetch = n_prefetch_;
    }
    mo.last_checkout_block_num = block_num_e - 1;
  }
  uint64_t nelems_pf = std::min(size + n_prefetch * block_size,
                                mo.size - ptr.offset()) / sizeof(T);

  section block_section{0, block_size};

  // get cache blocks first, otherwise the same cache block may be assigned to multiple mem blocks
  for_each_block(ptr, nelems_pf, [&](const auto& bi) {
    if (is_locally_accessible(bi.owner)) {
      bool is_prefetch = bi.offset_b >= ptr.offset() + size;
      if (!is_prefetch) {
        uint8_t* vm_addr = (uint8_t*)mo.vm.addr() + bi.offset_b;

        home_block& hb = std::invoke([&]() -> home_block& {
          try {
            return mmap_cache_.ensure_cached((uintptr_t)vm_addr / block_size);
          } catch (cache_full_exception& e) {
            die("mmap cache is exhausted (too many objects are being checked out)");
            throw;
          }
        });

        if (!hb.mapped) {
          hb.pm = &mo.home_pms[bi.owner];
          hb.pm_offset = bi.pm_offset;
          hb.vm_addr = vm_addr;
          hb.size = bi.offset_e - bi.offset_b;
          remap_home_blocks_.push_back(&hb);
        }

        hb.checkout_count++;
      }
    } else {
      // for each cache block
      uint64_t offset_b = std::max(bi.offset_b, ptr.offset() / block_size * block_size);
      uint64_t offset_e = std::min(bi.offset_e, ptr.offset() + nelems_pf * sizeof(T));
      for (uint64_t offset = offset_b; offset < offset_e; offset += block_size) {
        uint8_t* vm_addr = (uint8_t*)mo.vm.addr() + offset;

        cache_block& cb = std::invoke([&]() -> cache_block& {
          try {
            return cache_.ensure_cached((uintptr_t)vm_addr / block_size);
          } catch (cache_full_exception& e) {
            complete_flush();
            flush_dirty_cache();
            complete_flush();
            try {
              return cache_.ensure_cached((uintptr_t)vm_addr / block_size);
            } catch (cache_full_exception& e) {
              die("cache is exhausted (too many objects are being checked out)");
              throw;
            }
          }
        });

        if (!cb.mapped) {
          cb.vm_addr = vm_addr;
          cb.obj_id = mo.id;
          remap_cache_blocks_.push_back(&cb);
        }

        cb.transitive = true;

        bool is_prefetch = offset >= ptr.offset() + size;
        if (!is_prefetch) {
          cb.checkout_count++;
        }
      }
    }
  });

  // begin fetching data
  for_each_block(ptr, nelems_pf, [&](const auto& bi) {
    if (!is_locally_accessible(bi.owner)) {
      uint64_t offset_b = std::max(bi.offset_b, ptr.offset() / block_size * block_size);
      uint64_t offset_e = std::min(bi.offset_e, ptr.offset() + nelems_pf * sizeof(T));
      for (uint64_t offset = offset_b; offset < offset_e; offset += block_size) {
        uint8_t* vm_addr = (uint8_t*)mo.vm.addr() + offset;
        cache_block& cb = cache_.ensure_cached((uintptr_t)vm_addr / block_size);

        if (cb.flushing && Mode != access_mode::read) {
          // MPI_Put has been already started on this cache block.
          // As overlapping MPI_Put calls for the same location will cause undefined behaviour,
          // we need to insert MPI_Win_flush between overlapping MPI_Put calls here.
          auto ev2 = logger::template record<logger_kind::FlushConflicted>();
          complete_flush();
        }

        // If only a part of the block is written, we need to fetch the block
        // when this block is checked out again with read access mode.
        if (Mode == access_mode::write) {
          uint64_t offset_in_block_b = (ptr.offset() > offset) ?
                                       ptr.offset() - offset : 0;
          uint64_t offset_in_block_e = (ptr.offset() + size < offset + block_size) ?
                                       ptr.offset() + size - offset : block_size;
          sections_insert(cb.partial_sections, {offset_in_block_b, offset_in_block_e});
        }

        // Suppose that a cache block is represented as [a1, a2].
        // If a1 is checked out with write-only access mode, then [a1, a2] is allocated a cache entry,
        // but fetch for a1 and a2 is skipped.  Later, if a2 is checked out with read access mode,
        // the data for a2 would not be fetched because it is already in the cache.
        // Thus, we allocate a `partial` flag to each cache entry to indicate if the entire cache block
        // has been already fetched or not.
        if (Mode != access_mode::write) {
          fetch_begin(mo, cb, bi.owner, bi.pm_offset + offset - bi.offset_b);
          sections_insert(cb.partial_sections, block_section); // the entire cache block is now fetched
        }

        // cache with write-only mode is always valid
        if (Mode == access_mode::write &&
            cb.cstate != cache_state::fetching) {
          cb.cstate = cache_state::valid;
        }

        PCAS_CHECK(cb.transitive);
        cb.transitive = false;
      }
    }
  });

  // Overlap communication and memory remapping
  ensure_remapped();

  std::vector<MPI_Request> reqs;
  for_each_block(ptr, nelems, [&](const auto& bi) {
    if (!is_locally_accessible(bi.owner)) {
      uint64_t offset_b = std::max(bi.offset_b, ptr.offset() / block_size * block_size);
      uint64_t offset_e = std::min(bi.offset_e, ptr.offset() + size);
      for (uint64_t offset = offset_b; offset < offset_e; offset += block_size) {
        uint8_t* vm_addr = (uint8_t*)mo.vm.addr() + offset;
        cache_block& cb = cache_.ensure_cached((uintptr_t)vm_addr / block_size);

        if (cb.cstate == cache_state::fetching) {
          PCAS_CHECK(cb.req != MPI_REQUEST_NULL);
          reqs.push_back(cb.req);
          cb.req = MPI_REQUEST_NULL;
        }
        cb.cstate = cache_state::valid;
      }
    }
  });

  T* ret = (T*)((uint8_t*)mo.vm.addr() + ptr.offset());

  auto ckey = std::make_pair((void*)ret, size);
  auto c = checkouts_.find(ckey);
  if (c != checkouts_.end()) {
    // Only read-only access is allowed for overlapped checkout
    // TODO: dynamic check for conflicting access when the region is overlapped but not the same
    PCAS_CHECK(c->second.mode == access_mode::read);
    PCAS_CHECK(Mode == access_mode::read);
    c->second.count++;
  } else {
    checkouts_[ckey] = (checkout_entry){
      .ptr = static_cast<global_ptr<uint8_t>>(ptr), .mode = Mode, .count = 1,
    };
  }

  if (!reqs.empty()) {
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
  }

  return ret;
}

template <typename P>
template <typename T>
inline void pcas_if<P>::checkin(T* raw_ptr, uint64_t nelems) {
  uint64_t size = nelems * sizeof(T);
  auto ev = logger::template record<logger_kind::Checkin>();

  auto ckey = std::make_pair((void*)raw_ptr, size);
  auto c = checkouts_.find(ckey);
  if (c == checkouts_.end()) {
    die("The region [%p, %p) passed to checkin() is not registered", raw_ptr, raw_ptr + size);
  }

  checkout_entry che = c->second;
  mem_obj& mo = objs_[che.ptr.id()];

  for_each_block(che.ptr, size, [&](const auto& bi) {
    if (is_locally_accessible(bi.owner)) {
      uint8_t* vm_addr = (uint8_t*)mo.vm.addr() + bi.offset_b;
      home_block& hb = mmap_cache_.ensure_cached((uintptr_t)vm_addr / block_size);
      hb.checkout_count--;
    } else {
      uint64_t offset_b = std::max(bi.offset_b, che.ptr.offset() / block_size * block_size);
      uint64_t offset_e = std::min(bi.offset_e, che.ptr.offset() + size);
      for (uint64_t offset = offset_b; offset < offset_e; offset += block_size) {
        uint8_t* vm_addr = (uint8_t*)mo.vm.addr() + offset;
        cache_block& cb = cache_.ensure_cached((uintptr_t)vm_addr / block_size);

        if (che.mode != access_mode::read) {
          n_dirty_cache_blocks_ += cb.dirty_sections.empty();
          uint64_t offset_in_block_b = (che.ptr.offset() > offset) ?
                                       che.ptr.offset() - offset : 0;
          uint64_t offset_in_block_e = (che.ptr.offset() + size < offset + block_size) ?
                                       che.ptr.offset() + size - offset : block_size;
          sections_insert(cb.dirty_sections, {offset_in_block_b, offset_in_block_e});
          cache_dirty_ = true;
        }
        cb.checkout_count--;
      }
    }
  });

  if (--c->second.count == 0) {
    checkouts_.erase(ckey);
  }

  if (enable_write_through_ == 1) {
    // 1: wait for PUT completion here (strict write-through)
    ensure_all_cache_clean();
  } else if (enable_write_through_ == 2) {
    // 2: do not flush here (the next release waits for PUT completion)
    flush_dirty_cache();
  }
}

PCAS_TEST_CASE("[pcas::pcas] checkout and checkin (small, aligned)") {
  pcas pc;

  int rank = pc.rank();
  int nproc = pc.nproc();

  int n = pcas::block_size * nproc;
  global_ptr<uint8_t> ps[2];
  ps[0] = pc.malloc<uint8_t, mem_mapper::block >(n);
  ps[1] = pc.malloc<uint8_t, mem_mapper::cyclic>(n);

  for (auto p : ps) {
    uint8_t* home_ptr = (uint8_t*)pc.get_physical_mem(p);
    for (uint64_t i = 0; i < pcas::block_size; i++) {
      home_ptr[i] = rank;
    }

    pc.barrier();

    PCAS_SUBCASE("read the entire array") {
      const uint8_t* rp = pc.checkout<access_mode::read>(p, n);
      for (int i = 0; i < n; i++) {
        PCAS_CHECK_MESSAGE(rp[i] == i / pcas::block_size, "rank: ", rank, ", i: ", i);
      }
      pc.checkin(rp, n);
    }

    PCAS_SUBCASE("read and write the entire array") {
      for (int it = 0; it < nproc; it++) {
        if (it == rank) {
          uint8_t* rp = pc.checkout<access_mode::read_write>(p, n);
          for (int i = 0; i < n; i++) {
            PCAS_CHECK_MESSAGE(rp[i] == i / pcas::block_size + it, "it: ", it, ", rank: ", rank, ", i: ", i);
            rp[i]++;
          }
          pc.checkin(rp, n);
        }
        pc.barrier();

        const uint8_t* rp = pc.checkout<access_mode::read>(p, n);
        for (int i = 0; i < n; i++) {
          PCAS_CHECK_MESSAGE(rp[i] == i / pcas::block_size + it + 1, "it: ", it, ", rank: ", rank, ", i: ", i);
        }
        pc.checkin(rp, n);

        pc.barrier();
      }
    }

    PCAS_SUBCASE("read the partial array") {
      int ib = n / 5 * 2;
      int ie = n / 5 * 4;
      int s = ie - ib;

      const uint8_t* rp = pc.checkout<access_mode::read>(p + ib, s);
      for (int i = 0; i < s; i++) {
        PCAS_CHECK_MESSAGE(rp[i] == (i + ib) / pcas::block_size, "rank: ", rank, ", i: ", i);
      }
      pc.checkin(rp, s);
    }
  }

  pc.free(ps[0]);
  pc.free(ps[1]);
}

PCAS_TEST_CASE("[pcas::pcas] checkout and checkin (large, not aligned)") {
  pcas pc(16 * pcas::block_size);

  int rank = pc.rank();
  int nproc = pc.nproc();

  int n = 10000000;

  global_ptr<int> ps[2];
  ps[0] = pc.malloc<int, mem_mapper::block >(n);
  ps[1] = pc.malloc<int, mem_mapper::cyclic>(n);

  int max_checkout_size = (16 - 2) * pcas::block_size / sizeof(int);

  for (auto p : ps) {
    if (rank == 0) {
      for (int i = 0; i < n; i += max_checkout_size) {
        int m = std::min(max_checkout_size, n - i);
        int* rp = pc.checkout<access_mode::write>(p + i, m);
        for (int j = 0; j < m; j++) {
          rp[j] = i + j;
        }
        pc.checkin(rp, m);
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
        pc.checkin(rp, m);
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
        pc.checkin(rp, m);
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
        pc.checkin(rp, s);
      }

      pc.barrier();

      for (int i = 0; i < n; i += max_checkout_size) {
        int m = std::min(max_checkout_size, n - i);
        const int* rp = pc.checkout<access_mode::read>(p + i, m);
        for (int j = 0; j < m; j++) {
          PCAS_CHECK(rp[j] == (i + j) * 2);
        }
        pc.checkin(rp, m);
      }
    }
  }

  pc.free(ps[0]);
  pc.free(ps[1]);
}

// TODO: add tests to below functions

template <typename P>
inline void pcas_if<P>::release() {
  auto ev = logger::template record<logger_kind::Release>();
  ensure_all_cache_clean();
}

template <typename P>
inline void pcas_if<P>::release_lazy(release_handler* handler) {
  PCAS_CHECK(checkouts_.empty());

  epoch_t next_epoch = cache_dirty_ ? rm_.remote->epoch + 1 : 0; // 0 means clean
  *handler = {.rank = cg_global_.rank, .epoch = next_epoch};
}

template <typename P>
inline void pcas_if<P>::acquire(release_handler handler) {
  auto ev = logger::template record<logger_kind::Acquire>();
  ensure_all_cache_clean();

  if (handler.epoch != 0) {
    epoch_t remote_epoch = get_remote_epoch(handler.rank);
    if (remote_epoch < handler.epoch) {
      send_release_request(handler.rank, remote_epoch, handler.epoch);
      // need to wait for the execution of a release by the remote worker
      while (get_remote_epoch(handler.rank) < handler.epoch) {
        usleep(10); // TODO: better interval?
      };
    }
  }

  cache_invalidate_all();
}

template <typename P>
inline void pcas_if<P>::barrier() {
  release();
  MPI_Barrier(cg_global_.comm);
  acquire();
}

template <typename P>
inline void pcas_if<P>::poll() {
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

}
