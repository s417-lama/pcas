#pragma once

#include <type_traits>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <memory>

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
  using block_num_t = uint64_t;

  enum class cache_state {
    evicted,    // this entry is not in the cache
    invalid,    // this entry is in the cache, but the data is not up-to-date
    fetching,   // communication (read) is in-progress
    valid,      // the data is up-to-date
  };

  struct home_block {
    physical_mem&     pm;
    uint64_t          pm_offset;
    uint8_t*          vm_addr;
    uint64_t          size;
    int               checkout_count    = 0;
    bool              mapped            = false;
    cache_block_num_t block_num         = std::numeric_limits<block_num_t>::max();
    home_block*       prev_mapped_block = nullptr;

    home_block(physical_mem& pm,
               uint64_t pm_offset,
               uint8_t* vm_addr,
               uint64_t size) : pm(pm), pm_offset(pm_offset), vm_addr(vm_addr), size(size) {}

    void map() {
      pm.map(vm_addr, pm_offset, size);
    }

    void unmap() {
      virtual_mem::mmap_no_physical_mem(vm_addr, size);
    }

    /* Callback functions for cache_system class */

    bool is_evictable() const {
      return checkout_count == 0;
    }

    bool is_cached() const {
      return mapped;
    }

    cache_block_num_t get_cache_block_num() const {
      return block_num;
    }

    void on_cache_remap(cache_block_num_t b, home_block* prev_block) {
      block_num = b;
      prev_mapped_block = prev_block;
      mapped = true;
    }

    void on_evict() {
      PCAS_CHECK(is_evictable());
      block_num = std::numeric_limits<cache_block_num_t>::max();
      mapped = false;
    }
  };

  using mmap_cache_t = cache_system<home_block*>;

  struct mem_block {
    home_block*       home_blk          = nullptr;
    block_num_t       mem_block_num     = std::numeric_limits<block_num_t>::max();
    cache_block_num_t cache_block_num   = std::numeric_limits<block_num_t>::max();
    cache_state       cstate            = cache_state::evicted;
    bool              transitive        = false;
    bool              flushing          = false;
    int               checkout_count    = 0;
    uint8_t*          vm_addr           = nullptr;
    mem_block*        prev_cached_block = nullptr;
    MPI_Request       req               = MPI_REQUEST_NULL;
    obj_id_t          obj_id;
    sections          dirty_sections;
    sections          partial_sections; // for write-only update

    void invalidate() {
      if (cstate == cache_state::fetching) {
        PCAS_CHECK(req != MPI_REQUEST_NULL);
        // FIXME: MPI_Cancel causes segfault
        /* MPI_Cancel(&req); */
        /* MPI_Request_free(&req); */
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        PCAS_CHECK(req == MPI_REQUEST_NULL);
      }
      partial_sections.clear();
      cstate = cache_state::invalid;
      PCAS_CHECK(is_evictable());
    }

    /* Callback functions for cache_system class */

    bool is_evictable() const {
      return checkout_count == 0 && !transitive &&
             !flushing && dirty_sections.empty();
    }

    bool is_cached() const {
      return cstate != cache_state::evicted;
    }

    cache_block_num_t get_cache_block_num() const {
      return cache_block_num;
    }

    void on_cache_remap(cache_block_num_t b, mem_block* prev_block) {
      cache_block_num = b;
      prev_cached_block = prev_block;
      cstate = cache_state::invalid;
    }

    void on_evict() {
      PCAS_CHECK(is_evictable());
      invalidate();
      cache_block_num = std::numeric_limits<cache_block_num_t>::max();
      cstate = cache_state::evicted;
    }
  };

  using cache_t = cache_system<mem_block*>;

  struct mem_obj {
    int                                   owner;
    obj_id_t                              id;
    uint64_t                              size;
    uint64_t                              effective_size;
    std::unique_ptr<mem_mapper::base>     mmapper;
    std::unordered_map<int, physical_mem> home_pms;
    virtual_mem                           vm;
    std::vector<mem_block>                mem_blocks;
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
  };

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
  std::unordered_map<obj_id_t, mem_obj> objs_;

  // FIXME: using map instead of unordered_map because std::pair needs a user-defined hash function
  std::map<std::pair<void*, uint64_t>, checkout_entry> checkouts_;

  mmap_cache_t mmap_cache_;
  cache_t cache_;
  physical_mem cache_pm_;

  std::vector<home_block*> remap_home_blocks_;
  std::vector<mem_block*> remap_cache_blocks_;

  bool cache_dirty_ = false;
  release_manager rm_;

  std::unordered_set<MPI_Win> flushing_wins_;

  uint64_t n_dirty_cache_blocks_ = 0;
  uint64_t max_dirty_cache_blocks_;

  int enable_shared_memory_;

  int n_prefetch_;

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

  uint64_t get_home_mmap_limit(uint64_t n_cache_blocks) {
    uint64_t sys_limit = sys_mmap_entry_limit();
    uint64_t margin = 1000;
    PCAS_CHECK(sys_limit > n_cache_blocks + margin);
    return (sys_limit - n_cache_blocks - margin) / 2;
  }

  void ensure_remapped() {
    if (!remap_home_blocks_.empty()) {
      for (home_block* hb : remap_home_blocks_) {
        home_block* prev_hb = hb->prev_mapped_block;
        if (prev_hb) {
          hb->prev_mapped_block = nullptr;
          if (!prev_hb->is_cached()) {
            prev_hb->unmap();
          }
        }
        if (hb->is_cached()) {
          PCAS_CHECK(hb->block_num < mmap_cache_.num_blocks());
          hb->map();
        }
      }
      remap_home_blocks_.clear();
    }
    if (!remap_cache_blocks_.empty()) {
      for (mem_block* mb : remap_cache_blocks_) {
        mem_block* prev_mb = mb->prev_cached_block;
        if (prev_mb) {
          mb->prev_cached_block = nullptr;
          if (!prev_mb->is_cached()) {
            // save the number of mmap entries by unmapping previous virtual memory
            virtual_mem::mmap_no_physical_mem(prev_mb->vm_addr, block_size);
          }
        }
        if (mb->is_cached()) { // the block might have already been evicted if prefetched
          PCAS_CHECK(mb->cache_block_num < cache_.num_blocks());
          cache_pm_.map(mb->vm_addr, mb->cache_block_num * block_size, block_size);
        }
      }
      remap_cache_blocks_.clear();
    }
  }

  void fetch_begin(mem_block& mb, int owner, size_t owner_disp) {
    mem_obj& mo = objs_[mb.obj_id];
    void* cache_block_ptr = cache_pm_.anon_vm_addr();
    int count = 0;
    // fetch only nondirty sections
    for (auto [offset_in_block_b, offset_in_block_e] :
         sections_inverse(mb.partial_sections, {0, block_size})) {
      PCAS_CHECK(mb.cache_block_num < cache_.num_blocks());
      MPI_Request req;
      MPI_Rget((uint8_t*)cache_block_ptr + mb.cache_block_num * block_size + offset_in_block_b,
               offset_in_block_e - offset_in_block_b,
               MPI_UINT8_T,
               owner,
               owner_disp + offset_in_block_b,
               offset_in_block_e - offset_in_block_b,
               MPI_UINT8_T,
               mo.win,
               &req);
      mb.cstate = cache_state::fetching;
      // FIXME
      if (count == 0) {
        PCAS_CHECK(mb.req == MPI_REQUEST_NULL);
        mb.req = req;
      } else {
        MPI_Wait(&req, MPI_STATUS_IGNORE);
      }
      count++;
    }
  }

  void flush_dirty_cache() {
    if (cache_dirty_) {
      cache_.for_each_block([&](mem_block& mb) {
        if (!mb.dirty_sections.empty()) {
          PCAS_CHECK(!mb.flushing);

          mem_obj& mo = objs_[mb.obj_id];
          void* cache_block_ptr = cache_pm_.anon_vm_addr();
          uint64_t vm_offset = mb.vm_addr - (uint8_t*)mo.vm.addr();
          auto bi = mo.mmapper->get_block_info(vm_offset);

          for (auto [offset_in_block_b, offset_in_block_e] : mb.dirty_sections) {
            MPI_Put((uint8_t*)cache_block_ptr + mb.cache_block_num * block_size + offset_in_block_b,
                    offset_in_block_e - offset_in_block_b,
                    MPI_UINT8_T,
                    bi.owner,
                    bi.pm_offset + offset_in_block_b,
                    offset_in_block_e - offset_in_block_b,
                    MPI_UINT8_T,
                    mo.win);
          }
          mb.dirty_sections.clear();
          mb.flushing = true;

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

      cache_.for_each_block([&](mem_block& mb) {
        mb.flushing = false;
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
    cache_.for_each_block([&](mem_block& mb) {
      mb.invalidate();
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

public:
  using logger = typename logger::template logger_if<logger::policy<P>>;
  using logger_kind = typename P::logger_kind_t::value;

  constexpr static uint64_t block_size = P::block_size;

  pcas_if(uint64_t cache_size = 1024 * block_size, MPI_Comm comm = MPI_COMM_WORLD);
  ~pcas_if();

  int rank() const { return global_rank_; }
  int nproc() const { return global_nproc_; }

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
    return mo.home_pms[global_rank_].anon_vm_addr();
  }

};

template <typename P>
inline pcas_if<P>::pcas_if(uint64_t cache_size, MPI_Comm comm)
  : process_map_(init_process_map(comm)),
    intra2global_rank_(init_intra2global_rank()),
    mmap_cache_(get_home_mmap_limit(cache_size / block_size)),
    cache_(cache_size / block_size) {

  PCAS_CHECK(cache_size % block_size == 0);

  uint64_t pagesize = sysconf(_SC_PAGE_SIZE);
  if (block_size == 0 || block_size % pagesize != 0) {
    die("The block size (specified: %ld) must be multiple of the page size (%ld).", block_size, pagesize);
  }

  cache_pm_ = physical_mem(cache_size, 0, intra_rank_, true, true);

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
  enable_shared_memory_ = get_env("PCAS_ENABLE_SHARED_MEMORY", 1, global_rank_);
  n_prefetch_ = get_env("PCAS_PREFETCH_BLOCKS", 0, global_rank_);

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
template <typename T, template<uint64_t> typename MemMapper, typename... MemMapperArgs>
inline global_ptr<T> pcas_if<P>::malloc(uint64_t nelems, MemMapperArgs... mmargs) {
  if (nelems == 0) {
    die("nelems cannot be 0");
  }

  uint64_t size = nelems * sizeof(T);

  std::unique_ptr<MemMapper<block_size>> mmapper(
    new MemMapper<block_size>(size, global_nproc_, mmargs...));

  uint64_t local_size = mmapper->get_local_size(global_rank_);
  uint64_t effective_size = mmapper->get_effective_size();

  obj_id_t obj_id = obj_id_count_++;

  virtual_mem vm(nullptr, effective_size);
  physical_mem pm_local(local_size, obj_id, intra_rank_, true, true);

  MPI_Win win = MPI_WIN_NULL;
  MPI_Win_create(pm_local.anon_vm_addr(),
                 local_size,
                 1,
                 MPI_INFO_NULL,
                 global_comm_,
                 &win);
  MPI_Win_lock_all(0, win);

  // Open home physical memory of other intra-node processes
  std::unordered_map<int, physical_mem> home_pms;
  for (int i = 0; i < intra_nproc_; i++) {
    if (i == intra_rank_) {
      home_pms[global_rank_] = std::move(pm_local);
    } else if (enable_shared_memory_) {
      int target_rank = intra2global_rank_[i];
      int target_local_size = mmapper->get_local_size(target_rank);
      physical_mem pm(target_local_size, obj_id, i, false, false);
      home_pms[target_rank] = std::move(pm);
    }
  }

  // Create memory blocks
  std::vector<mem_block> mem_blocks;
  for (block_num_t b = 0; b < effective_size / block_size; b++) {
    mem_block mb;
    mb.obj_id = obj_id;
    mb.mem_block_num = b;
    mb.vm_addr = (uint8_t*)vm.addr() + b * block_size;
    mem_blocks.push_back(std::move(mb));
  }

  mem_obj mo {
    .owner = -1, .id = obj_id,
    .size = size, .effective_size = effective_size,
    .mmapper = std::move(mmapper),
    .home_pms = std::move(home_pms), .vm = std::move(vm),
    .mem_blocks = std::move(mem_blocks),
    .last_checkout_block_num = std::numeric_limits<block_num_t>::max(),
    .win = win,
  };

  auto ret = global_ptr<T>(mo.owner, mo.id, 0);

  objs_[mo.id] = std::move(mo);

  // Create entries for home blocks
  for_each_block(ret, nelems, [&](int      owner,
                                  uint64_t offset_b,
                                  uint64_t offset_e,
                                  uint64_t pm_offset) {
    PCAS_CHECK(offset_b % block_size == 0);
    if (owner == global_rank_ ||
        (enable_shared_memory_ && process_map_[owner].second == inter_rank_)) {
      mem_obj& mo = objs_[obj_id];
      home_block* hb = new home_block(mo.home_pms[owner],
                                      pm_offset,
                                      (uint8_t*)mo.vm.addr() + offset_b,
                                      offset_e - offset_b);
      // Register home block for each memory block within the home block
      // (Note: home blocks can be larger than memory blocks; cf. block dist policy)
      for (block_num_t b = offset_b / block_size; b < offset_e / block_size; b++) {
        mo.mem_blocks[b].home_blk = hb;
      }
    }
  });

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
  for (auto&& mb : mo.mem_blocks) {
    if (mb.home_blk) {
      mmap_cache_.ensure_evicted(mb.home_blk);
    } else {
      cache_.ensure_evicted(&mb);
    }
  }

  // free home blocks
  size_t nelems = mo.size / sizeof(T);
  for_each_block(ptr, nelems, [&](int      owner,
                                  uint64_t offset_b,
                                  uint64_t offset_e [[maybe_unused]],
                                  uint64_t pm_offset [[maybe_unused]]) {
    if (owner == global_rank_ ||
        (enable_shared_memory_ && process_map_[owner].second == inter_rank_)) {
      // Only the home block pointed by the first memory block should be freed.
      // Home blocks can be pointed by multiple memory blocks.
      delete mo.mem_blocks[offset_b / block_size].home_blk;
    }
  });

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
    uint64_t offset_b  = std::max(bi.offset_b, offset_min);
    uint64_t offset_e  = std::min(bi.offset_e, offset_max);

    fn(bi.owner, offset_b, offset_e, bi.pm_offset);

    offset = bi.offset_e;
  }
}

PCAS_TEST_CASE("[pcas::pcas] loop over blocks") {
  pcas pc;

  int nproc = pc.nproc();

  int n = 100000;
  auto p = pc.malloc<int, mem_mapper::block>(n);

  PCAS_SUBCASE("loop over the entire array") {
    int prev_owner = -1;
    uint64_t prev_offset_e = 0;
    pc.for_each_block(p, n, [&](int      owner,
                                uint64_t offset_b,
                                uint64_t offset_e,
                                uint64_t pm_offset [[maybe_unused]]) {
      PCAS_CHECK(owner == prev_owner + 1);
      PCAS_CHECK(offset_b == prev_offset_e);
      prev_owner = owner;
      prev_offset_e = offset_e;
    });
    PCAS_CHECK(prev_owner == nproc - 1);
    PCAS_CHECK(prev_offset_e == n * sizeof(int));
  }

  PCAS_SUBCASE("loop over the partial array") {
    int b = n / 5 * 2;
    int e = n / 5 * 4;
    int s = e - b;

    mem_mapper::block<pcas::block_size> mmapper{n * sizeof(int), nproc};
    int o1 = mmapper.get_block_info(b * sizeof(int)).owner;
    int o2 = mmapper.get_block_info(e * sizeof(int)).owner;

    int prev_owner = o1 - 1;
    uint64_t prev_offset_e = b * sizeof(int);
    pc.for_each_block(p + b, s, [&](int      owner,
                                    uint64_t offset_b,
                                    uint64_t offset_e,
                                    uint64_t pm_offset [[maybe_unused]]) {
      PCAS_CHECK(owner == prev_owner + 1);
      PCAS_CHECK(offset_b == prev_offset_e);
      prev_owner = owner;
      prev_offset_e = offset_e;
    });
    auto& o2_ = o2; // structured bindings cannot be captured by lambda until C++20
    PCAS_CHECK(prev_owner == o2_);
    PCAS_CHECK(prev_offset_e == e * sizeof(int));
  }

  pc.free(p);
}

template <typename P>
template <typename T>
inline void pcas_if<P>::get(global_ptr<T> from_ptr, T* to_ptr, uint64_t nelems) {
  if (from_ptr.owner() == -1) {
    mem_obj& mo = objs_[from_ptr.id()];
    uint64_t offset = from_ptr.offset();
    std::vector<MPI_Request> reqs;

    for_each_block(from_ptr, nelems, [&](int      owner,
                                         uint64_t offset_b,
                                         uint64_t offset_e,
                                         uint64_t pm_offset) {
      MPI_Request req;
      MPI_Rget((uint8_t*)to_ptr - offset + offset_b,
               offset_e - offset_b,
               MPI_UINT8_T,
               owner,
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
    uint64_t offset = to_ptr.offset();

    for_each_block(to_ptr, nelems, [&](int      owner,
                                       uint64_t offset_b,
                                       uint64_t offset_e,
                                       uint64_t pm_offset) {
      MPI_Put((uint8_t*)from_ptr - offset + offset_b,
              offset_e - offset_b,
              MPI_UINT8_T,
              owner,
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

  block_num_t block_num_b = ptr.offset() / block_size;
  block_num_t block_num_e = (ptr.offset() + size + block_size - 1) / block_size;

  section block_section{0, block_size};

  block_num_t bmax = block_num_e;

  // get cache blocks first, otherwise the same cache block may be assigned to multiple mem blocks
  for (block_num_t b = block_num_b; b < bmax; b++) {
    auto& mb = mo.mem_blocks[b];
    if (!mb.home_blk) {
      cache_state prev_cstate = mb.cstate;

      try {
        cache_.ensure_cached(&mb);
      } catch (cache_full_exception& e) {
        // If the cache is full, just quit prefetching
        bmax = b;
        break;
      }

      if (prev_cstate == cache_state::evicted) {
        remap_cache_blocks_.push_back(&mb);
      }

      mb.transitive = true;
    }
  }

  for (block_num_t b = block_num_b; b < bmax; b++) {
    auto& mb = mo.mem_blocks[b];
    if (!mb.is_home) {
      auto bi = mo.mmapper->get_block_info(b * block_size);

      fetch_begin(mb, bi.owner, bi.pm_offset);

      sections_insert(mb.partial_sections, block_section);

      PCAS_CHECK(mb.transitive);
      mb.transitive = false;
    }
  }
}

template <typename P>
template <access_mode Mode, typename T>
inline std::conditional_t<Mode == access_mode::read, const T*, T*>
pcas_if<P>::checkout(global_ptr<T> ptr, uint64_t nelems) {
  uint64_t size = nelems * sizeof(T);
  auto ev = logger::template record<logger_kind::Checkout>(size);

  mem_obj& mo = objs_[ptr.id()];

  block_num_t block_num_b = ptr.offset() / block_size;
  block_num_t block_num_e = (ptr.offset() + size + block_size - 1) / block_size;

  uint64_t bmax = mo.effective_size / block_size;

  int n_prefetch = 0;
  if (Mode != access_mode::write) {
    if (block_num_b <= mo.last_checkout_block_num + 1 &&
        mo.last_checkout_block_num + 1 < block_num_e) {
      // If it seems sequential access, do prefetch
      n_prefetch = n_prefetch_;
    }
    mo.last_checkout_block_num = block_num_e - 1;
  }

  section block_section{0, block_size};

  // get cache blocks first, otherwise the same cache block may be assigned to multiple mem blocks
  for (block_num_t b = block_num_b; b < std::min(block_num_e + n_prefetch, bmax); b++) {
    bool is_prefetch = b >= block_num_e;
    auto& mb = mo.mem_blocks[b];
    if (mb.home_blk) {
      // no prefetch for home blocks
      if (is_prefetch) continue;

      home_block& hb = *mb.home_blk;
      bool prev_mapped = hb.mapped;

      try {
        mmap_cache_.ensure_cached(&hb);
      } catch (cache_full_exception& e) {
        die("mmap cache is exhausted (too many objects are being checked out)");
      }

      if (!prev_mapped) {
        remap_home_blocks_.push_back(&hb);
      }

      hb.checkout_count++;
    } else {
      cache_state prev_cstate = mb.cstate;

      try {
        cache_.ensure_cached(&mb);
      } catch (cache_full_exception& e) {
        complete_flush();
        flush_dirty_cache();
        complete_flush();
        try {
          cache_.ensure_cached(&mb);
        } catch (cache_full_exception& e) {
          die("cache is exhausted (too many objects are being checked out)");
        }
      }

      if (prev_cstate == cache_state::evicted) {
        remap_cache_blocks_.push_back(&mb);
      }

      mb.transitive = true;

      if (!is_prefetch) {
        mb.checkout_count++;
      }
    }
  }

  for (block_num_t b = block_num_b; b < std::min(block_num_e + n_prefetch, bmax); b++) {
    auto& mb = mo.mem_blocks[b];
    if (!mb.home_blk) {
      if (mb.flushing && Mode != access_mode::read) {
        // MPI_Put has been already started on this cache block.
        // As overlapping MPI_Put calls for the same location will cause undefined behaviour,
        // we need to insert MPI_Win_flush between overlapping MPI_Put calls here.
        auto ev2 = logger::template record<logger_kind::FlushConflicted>();
        complete_flush();
      }

      // If only a part of the block is written, we need to fetch the block
      // when this block is checked out again with read access mode.
      if (Mode == access_mode::write) {
        uint64_t offset_in_block_b = (ptr.offset() > b * block_size) ?
                                     ptr.offset() - b * block_size : 0;
        uint64_t offset_in_block_e = (ptr.offset() + size < (b + 1) * block_size) ?
                                     ptr.offset() + size - b * block_size : block_size;
        sections_insert(mb.partial_sections, {offset_in_block_b, offset_in_block_e});
      }

      // Suppose that a cache block is represented as [a1, a2].
      // If a1 is checked out with write-only access mode, then [a1, a2] is allocated a cache entry,
      // but fetch for a1 and a2 is skipped.  Later, if a2 is checked out with read access mode,
      // the data for a2 would not be fetched because it is already in the cache.
      // Thus, we allocate a `partial` flag to each cache entry to indicate if the entire cache block
      // has been already fetched or not.
      if (Mode != access_mode::write) {
        auto bi = mo.mmapper->get_block_info(b * block_size);

        fetch_begin(mb, bi.owner, bi.pm_offset);

        sections_insert(mb.partial_sections, block_section); // the entire cache block is now fetched
      }

      // cache with write-only mode is always valid
      if (Mode == access_mode::write &&
          mb.cstate != cache_state::fetching) {
        mb.cstate = cache_state::valid;
      }

      PCAS_CHECK(mb.transitive);
      mb.transitive = false;
    }
  }

  // Overlap communication and memory remapping
  ensure_remapped();

  std::vector<MPI_Request> reqs;
  for (uint64_t b = block_num_b; b < block_num_e; b++) {
    auto& mb = mo.mem_blocks[b];
    if (!mb.home_blk) {
      if (mb.cstate == cache_state::fetching) {
        PCAS_CHECK(mb.req != MPI_REQUEST_NULL);
        reqs.push_back(mb.req);
        mb.req = MPI_REQUEST_NULL;
      }
      mb.cstate = cache_state::valid;
      /* cache_.use(&mb); */
    }
  }

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

  block_num_t block_num_b = che.ptr.offset() / block_size;
  block_num_t block_num_e = (che.ptr.offset() + size + block_size - 1) / block_size;

  for (block_num_t b = block_num_b; b < block_num_e; b++) {
    auto& mb = mo.mem_blocks[b];
    if (mb.home_blk) {
      home_block& hb = *mb.home_blk;
      hb.checkout_count--;
    } else {
      if (che.mode != access_mode::read) {
        n_dirty_cache_blocks_ += mb.dirty_sections.empty();
        uint64_t offset_in_block_b = (che.ptr.offset() > b * block_size) ?
                                     che.ptr.offset() - b * block_size : 0;
        uint64_t offset_in_block_e = (che.ptr.offset() + size < (b + 1) * block_size) ?
                                     che.ptr.offset() + size - b * block_size : block_size;
        sections_insert(mb.dirty_sections, {offset_in_block_b, offset_in_block_e});
        cache_dirty_ = true;
      }
      mb.checkout_count--;
    }
  }

  if (--c->second.count == 0) {
    checkouts_.erase(ckey);
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
  *handler = {.rank = global_rank_, .epoch = next_epoch};
}

template <typename P>
inline void pcas_if<P>::acquire(release_handler handler) {
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

  cache_invalidate_all();
}

template <typename P>
inline void pcas_if<P>::barrier() {
  release();
  MPI_Barrier(global_comm_);
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
