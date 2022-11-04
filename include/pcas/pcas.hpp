#pragma once

#include <cstring>
#include <type_traits>
#include <memory>
#include <functional>
#include <optional>

#include <mpi.h>

#include "pcas/util.hpp"
#include "pcas/global_ptr.hpp"
#include "pcas/physical_mem.hpp"
#include "pcas/virtual_mem.hpp"
#include "pcas/cache.hpp"
#include "pcas/wallclock.hpp"
#include "pcas/logger/logger.hpp"
#include "pcas/mem_mapper.hpp"
#include "pcas/allocator.hpp"
#include "pcas/topology.hpp"
#include "pcas/mem_obj.hpp"

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
  template <typename GPtrT>
  using global_ref = global_ref_base<GPtrT>;
  using wallclock_t = wallclock_native;
  using logger_kind_t = logger::kind;
  template <typename P>
  using logger_impl_t = logger::impl_dummy<P>;
  template <typename P>
  using allocator_impl_t = std_pool_resource_impl<P>;
  template <uint64_t BlockSize>
  using default_mem_mapper = mem_mapper::cyclic<BlockSize>;
  constexpr static uint64_t block_size = 65536;
  constexpr static bool enable_write_through = false;
};

template <typename P>
class pcas_if;

using pcas = pcas_if<policy_default>;

template <typename P>
class pcas_if {
  using this_t = pcas_if<P>;

  // Policies
  // -----------------------------------------------------------------------------

  struct global_ptr_policy {
    template <typename GPtrT>
    using global_ref = typename P::template global_ref<GPtrT>;
  };
  template <typename T>
  using global_ptr_ = global_ptr_if<global_ptr_policy, T>;

  struct logger_policy {
    static const char* outfile_prefix() { return "pcas"; }
    using wallclock_t = typename P::wallclock_t;
    using logger_kind_t = typename P::logger_kind_t;
    template <typename P_>
    using logger_impl_t = typename P::template logger_impl_t<P_>;
  };
  using logger_ = typename logger::template logger_if<logger_policy>;

  struct allocator_policy {
    constexpr static uint64_t block_size = P::block_size;
    using logger = logger_;
    template <typename P_>
    using allocator_impl_t = typename P::template allocator_impl_t<P_>;
  };
  using allocator = allocator_if<allocator_policy>;

  struct mem_obj_policy {
    constexpr static uint64_t block_size = P::block_size;
  };
  using mem_obj = mem_obj_if<mem_obj_policy>;

public:
  template <typename T>
  using global_ptr = global_ptr_<T>;
  using logger = logger_;
  using logger_kind = typename logger::kind::value;

private:

  // cache block
  // -----------------------------------------------------------------------------

  using cache_key_t = uintptr_t;

  enum class cache_state {
    unmapped,   // initial state
    invalid,    // this entry is in the cache, but the data is not up-to-date
    fetching,   // communication (read) is in-progress
    valid,      // the data is up-to-date
  };

  struct cache_block {
    cache_entry_num_t entry_num      = std::numeric_limits<cache_entry_num_t>::max();
    cache_state       cstate         = cache_state::unmapped;
    uint64_t          transaction_id = 0;
    bool              flushing       = false;
    int               checkout_count = 0;
    uint8_t*          vm_addr        = nullptr;
    uint8_t*          prev_vm_addr   = nullptr;
    bool              mapped         = false;
    topology::rank_t  owner          = -1;
    uint64_t          pm_offset      = 0;
    MPI_Win           win            = MPI_WIN_NULL;
    MPI_Request       req            = MPI_REQUEST_NULL;
    sections          dirty_sections;
    sections          partial_sections; // for write-only update
    this_t&           outer;

    cache_block(this_t& outer_ref) : outer(outer_ref) {}

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

    void map(const physical_mem& cache_pm) {
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
      return checkout_count == 0 && transaction_id != outer.transaction_id_ &&
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

      // for safety
      outer.cache_tlb_.clear();
    }

    void on_cache_map(cache_entry_num_t b) {
      entry_num = b;
      cstate = cache_state::invalid;
    }
  };

  using cache_t = cache_system<cache_key_t, cache_block>;

  // mmap cache block
  // -----------------------------------------------------------------------------

  struct home_block {
    cache_entry_num_t   entry_num = std::numeric_limits<cache_entry_num_t>::max();
    const physical_mem* pm;
    uint64_t            pm_offset;
    uint8_t*            vm_addr;
    uint64_t            size;
    uint64_t            transaction_id = 0;
    int                 checkout_count = 0;
    uint8_t*            prev_vm_addr   = nullptr;
    bool                mapped         = false;
    this_t&             outer;

    home_block(this_t& outer_ref) : outer(outer_ref) {}

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
      return checkout_count == 0 && transaction_id != outer.transaction_id_;
    }

    void on_evict() {
      PCAS_CHECK(is_evictable());
      PCAS_CHECK(prev_vm_addr == nullptr);
      prev_vm_addr = (uint8_t*)vm_addr;
      mapped = false;

      // for safety
      outer.home_tlb_.clear();
    }

    void on_cache_map(cache_entry_num_t b) {
      entry_num = b;
    }
  };

  using mmap_cache_t = cache_system<cache_key_t, home_block>;

  // TLB
  // -----------------------------------------------------------------------------

  template <typename T, int MaxEntries = 3>
  class tlb {
    std::array<std::optional<T>, MaxEntries> entries_;
    int last_index_ = 0;

  public:
    template <typename Fn>
    std::optional<T> find(Fn&& f) {
      for (auto& e : entries_) {
        if (e.has_value()) {
          bool found = std::forward<Fn>(f)(*e);
          if (found) {
            return *e;
          }
        }
      }
      return std::nullopt;
    }

    void add(T entry) {
      int idx = last_index_ % MaxEntries;
      entries_[idx] = entry;
      last_index_++;
    }

    void clear() {
      for (auto& e : entries_) {
        e.reset();
      }
    }
  };

  // cache management
  // -----------------------------------------------------------------------------

  struct checkout_entry {
    global_ptr<uint8_t> ptr;
    void*               raw_ptr;
    uint64_t            size;
    access_mode         mode;
    uint64_t            count;
  };

  struct release_remote_region {
    epoch_t request; // requested epoch from remote
    epoch_t epoch; // current epoch of the owner
  };

  class release_manager {
    win_manager win_;
    release_remote_region* remote_;

  public:
    release_manager(MPI_Comm comm) : win_(comm, sizeof(release_remote_region), (void**)&remote_) {
      remote_->request = 1;
      remote_->epoch = 1;
    }

    MPI_Win win() const { return win_.win(); }

    epoch_t request() const { return remote_->request; }
    epoch_t epoch() const { return remote_->epoch; }

    void increment_epoch() {
      remote_->epoch++;
    }
  };

  // Member variables
  // -----------------------------------------------------------------------------

  bool validate_dummy_; // for initialization

  topology topo_;

  std::vector<std::optional<mem_obj>> objs_;

  std::vector<std::optional<checkout_entry>> checkouts_;

  mmap_cache_t mmap_cache_;
  cache_t cache_;
  physical_mem cache_pm_;

  std::vector<home_block*> remap_home_blocks_;
  std::vector<cache_block*> remap_cache_blocks_;

  tlb<home_block*> home_tlb_;
  tlb<cache_block*> cache_tlb_;

  allocator allocator_;

  uint64_t transaction_id_ = 1;

  bool cache_dirty_ = false;
  release_manager rm_;

  std::vector<MPI_Win> flushing_wins_;
  bool early_flushing_ = false;

  uint64_t max_dirty_cache_blocks_;
  std::vector<cache_block*> dirty_cache_blocks_;

  // Initializaiton
  // -----------------------------------------------------------------------------

  bool validate_input(uint64_t cache_size, MPI_Comm comm) const {
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

  uint64_t calc_home_mmap_limit(uint64_t n_cache_blocks) const {
    uint64_t sys_limit = sys_mmap_entry_limit();
    uint64_t margin = 1000;
    PCAS_CHECK(sys_limit > n_cache_blocks + margin);
    return (sys_limit - n_cache_blocks - margin) / 2;
  }

  std::string cache_shmem_name(int global_rank) {
    std::stringstream ss;
    ss << "/pcas_cache_" << global_rank;
    return ss.str();
  }

  // Checkout entry
  // -----------------------------------------------------------------------------

  std::optional<checkout_entry>* find_checkout_entry(void* raw_ptr, uint64_t size) {
    for (auto& c : checkouts_) {
      if (c.has_value() && c->raw_ptr == raw_ptr && c->size == size) {
        return &c;
      }
    }
    return nullptr;
  }

  void add_checkout_entry(checkout_entry che) {
    for (auto& c : checkouts_) {
      if (!c.has_value()) {
        c = che;
        return;
      }
    }
    checkouts_.emplace_back(che);
  }

  void remove_checkout_entry(std::optional<checkout_entry>* c) {
    PCAS_CHECK(c);
    c->reset();
  }

  // Misc functions
  // -----------------------------------------------------------------------------

  cache_key_t cache_key(void* vm_addr) {
    PCAS_CHECK(reinterpret_cast<uintptr_t>(vm_addr) % block_size == 0);
    return reinterpret_cast<uintptr_t>(vm_addr) / block_size;
  }

  mem_obj& get_mem_obj(mem_obj_id_t id) {
    PCAS_CHECK(objs_[id].has_value());
    return *objs_[id];
  }

  // Memory mapping
  // -----------------------------------------------------------------------------

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

  home_block& ensure_mmap_cached_strict(void* vm_addr) {
    try {
      return mmap_cache_.ensure_cached(cache_key(vm_addr));
    } catch (cache_full_exception& e) {
      die("mmap cache is exhausted (too many objects are being checked out)");
      throw;
    }
  }

  cache_block& ensure_cached_strict(void* vm_addr) {
    try {
      return cache_.ensure_cached(cache_key(vm_addr));
    } catch (cache_full_exception& e) {
      ensure_all_cache_clean();
      try {
        return cache_.ensure_cached(cache_key(vm_addr));
      } catch (cache_full_exception& e) {
        die("cache is exhausted (too many objects are being checked out)");
        throw;
      }
    }
  }

  // Cached data management
  // -----------------------------------------------------------------------------

  void fetch_begin(cache_block& cb) {
    PCAS_CHECK(cb.owner < topo_.global_nproc());

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
               cb.owner,
               cb.pm_offset + offset_in_block_b,
               offset_in_block_e - offset_in_block_b,
               MPI_UINT8_T,
               cb.win,
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

    sections_insert(cb.partial_sections, {0, block_size}); // the entire cache block is now fetched
  }

  void add_dirty_cache_block(cache_block& cb) {
    dirty_cache_blocks_.push_back(&cb);
    cache_dirty_ = true;

    if (P::enable_write_through) {
      flush_dirty_cache_block(cb);
    } else if (dirty_cache_blocks_.size() >= max_dirty_cache_blocks_) {
      auto ev = logger::template record<logger_kind::FlushEarly>();
      flush_dirty_cache();
      dirty_cache_blocks_.clear();
      early_flushing_ = true;
    }
  }

  void flush_dirty_cache_block(cache_block& cb) {
    PCAS_CHECK(!cb.flushing);

    void* cache_block_ptr = cache_pm_.anon_vm_addr();

    for (auto [offset_in_block_b, offset_in_block_e] : cb.dirty_sections) {
      MPI_Put((uint8_t*)cache_block_ptr + cb.entry_num * block_size + offset_in_block_b,
              offset_in_block_e - offset_in_block_b,
              MPI_UINT8_T,
              cb.owner,
              cb.pm_offset + offset_in_block_b,
              offset_in_block_e - offset_in_block_b,
              MPI_UINT8_T,
              cb.win);
    }
    cb.dirty_sections.clear();
    cb.flushing = true;

    auto w = std::find(flushing_wins_.begin(), flushing_wins_.end(), cb.win);
    if (w == flushing_wins_.end()) {
      flushing_wins_.push_back(cb.win);
    }
  }

  void flush_dirty_cache() {
    for (auto& cb : dirty_cache_blocks_) {
      if (!cb->dirty_sections.empty()) {
        flush_dirty_cache_block(*cb);
      }
    }
  }

  void complete_flush() {
    if (!flushing_wins_.empty()) {
      if (early_flushing_) {
        // When early flush happened, dirty_cache_blocks_ was cleared, so we cannot make
        // assumption on which cache blocks are possibly flushing here
        cache_.for_each_entry([&](cache_block& cb) {
          cb.flushing = false;
        });
        early_flushing_ = false;
      } else {
        // We can reduce the number of iterations if early flush did not happen
        for (auto& cb : dirty_cache_blocks_) {
          cb->flushing = false;
        }
      }

      for (auto win : flushing_wins_) {
        MPI_Win_flush_all(win);
      }
      flushing_wins_.clear();
    }
  }

  void ensure_all_cache_clean() {
    PCAS_CHECK(empty(checkouts_));
    flush_dirty_cache();
    complete_flush();

    dirty_cache_blocks_.clear();

    if (cache_dirty_) {
      cache_dirty_ = false;
      rm_.increment_epoch();
    }
  }

  void cache_invalidate_all() {
    cache_.for_each_entry([&](cache_block& cb) {
      cb.invalidate();
    });
    cache_tlb_.clear();
  }

  // Lazy release
  // -----------------------------------------------------------------------------

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
             rm_.win(),
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
                           rm_.win());
      MPI_Win_flush(target_rank, rm_.win());
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
    /*                  rm_.win()); */
    /* MPI_Win_flush(target_rank, rm_.win()); */
  }

  // Checkout/checkin impl
  // -----------------------------------------------------------------------------

  template <access_mode Mode, bool DoCheckout>
  void* checkout_impl(global_ptr<uint8_t> ptr, uint64_t size);

  template <access_mode Mode, bool DoCheckout>
  void* checkout_impl_local(global_ptr<uint8_t> ptr, uint64_t size);

  template <bool DoCheckout>
  void checkin_impl(global_ptr<uint8_t> ptr, uint64_t size, access_mode mode);

  template <bool DoCheckout>
  void checkin_impl_local(global_ptr<uint8_t> ptr, uint64_t size, access_mode mode);

public:
  constexpr static uint64_t block_size = P::block_size;

  pcas_if(uint64_t cache_size = 1024 * block_size, MPI_Comm comm = MPI_COMM_WORLD);

  int rank() const { return topo_.global_rank(); }
  int nproc() const { return topo_.global_nproc(); }

  template <typename T>
  global_ptr<T> malloc(uint64_t nelems);

  template <typename T, template <uint64_t> typename MemMapper, typename... MemMapperArgs>
  global_ptr<T> malloc(uint64_t nelems, MemMapperArgs... mmargs);

  template <typename T>
  global_ptr<T> malloc_local(uint64_t nelems);

  template <typename T>
  void free(global_ptr<T> ptr);

  template <typename T, typename Func>
  void for_each_block(global_ptr<T> ptr, uint64_t nelems, Func fn);

  template <typename ConstT, typename T>
  std::enable_if_t<std::is_same_v<std::remove_const_t<ConstT>, T>>
  get(global_ptr<ConstT> from_ptr, T* to_ptr, uint64_t nelems);

  template <typename T>
  void put(const T* from_ptr, global_ptr<T> to_ptr, uint64_t nelems);

  template <typename ConstT, typename T>
  std::enable_if_t<std::is_same_v<std::remove_const_t<ConstT>, T>>
  get_nocache(global_ptr<ConstT> from_ptr, T* to_ptr, uint64_t nelems);

  template <typename T>
  void put_nocache(const T* from_ptr, global_ptr<T> to_ptr, uint64_t nelems);

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
    mem_obj& mo = get_mem_obj(ptr.id());
    return mo.home_pm().anon_vm_addr();
  }

};

template <typename P>
inline pcas_if<P>::pcas_if(uint64_t cache_size, MPI_Comm comm)
  : validate_dummy_(validate_input(cache_size, comm)),
    topo_(comm),
    objs_(1), // objs_[0] = dummy
    mmap_cache_(calc_home_mmap_limit(cache_size / block_size), *this),
    cache_(cache_size / block_size, *this),
    cache_pm_(cache_shmem_name(rank()), cache_size, true, true),
    allocator_(topo_),
    rm_(comm),
    max_dirty_cache_blocks_(get_env("PCAS_MAX_DIRTY_CACHE_SIZE", cache_size / 4, rank()) / block_size) {

  logger::init(rank(), nproc());

  dirty_cache_blocks_.reserve(max_dirty_cache_blocks_);

  barrier();
}

PCAS_TEST_CASE("[pcas::pcas] initialize and finalize PCAS") {
  for (int i = 0; i < 3; i++) {
    pcas pc;
  }
}

template <typename P>
template <typename T>
inline typename pcas_if<P>::template global_ptr<T>
pcas_if<P>::malloc(uint64_t nelems) {
  return malloc<T, P::template default_mem_mapper>(nelems);
}

template <typename P>
template <typename T, template <uint64_t> typename MemMapper, typename... MemMapperArgs>
inline typename pcas_if<P>::template global_ptr<T>
pcas_if<P>::malloc(uint64_t nelems, MemMapperArgs... mmargs) {
  if (nelems == 0) {
    die("nelems cannot be 0");
  }

  uint64_t size = nelems * sizeof(T);
  mem_obj_id_t obj_id = objs_.size();

  auto mmapper = std::make_unique<MemMapper<block_size>>(size, nproc(), mmargs...);

  objs_.emplace_back(std::in_place, std::move(mmapper), obj_id, size, topo_);

  return global_ptr<T>(-1, obj_id, 0);
}

template <typename P>
template <typename T>
inline typename pcas_if<P>::template global_ptr<T>
pcas_if<P>::malloc_local(uint64_t nelems) {
  if (nelems == 0) {
    die("nelems cannot be 0");
  }

  uint64_t size = nelems * sizeof(T);
  void* raw_ptr = allocator_.allocate(size);

  // FIXME: workaround: treat ptr.id() as size
  return global_ptr<T>(rank(), size, reinterpret_cast<uint64_t>(raw_ptr));
}

template <typename P>
template <typename T>
inline void pcas_if<P>::free(global_ptr<T> ptr) {
  if (ptr == global_ptr<T>()) {
    die("null pointer was passed to pcas::free()");
  }

  if (ptr.owner() == -1) { // Collective memory object
    // ensure free safety
    ensure_remapped();
    ensure_all_cache_clean();

    mem_obj& mo = get_mem_obj(ptr.id());

    // ensure all cache entries are evicted
    for (uint64_t offset = 0; offset < mo.effective_size(); offset += block_size) {
      uint8_t* vm_addr = reinterpret_cast<uint8_t*>(mo.vm().addr()) + offset;
      mmap_cache_.ensure_evicted(cache_key(vm_addr));
      cache_.ensure_evicted(cache_key(vm_addr));
    }

    home_tlb_.clear();
    cache_tlb_.clear();

    objs_[ptr.id()].reset();

  } else if (ptr.owner() == rank()) { // owned local objects
    // FIXME: workaround: treat ptr.id() as size
    allocator_.deallocate(reinterpret_cast<void*>(ptr.offset()), ptr.id());

  } else { // nonowned local objects
    // ensure the dirty cache of this memory object is discarded
    const uint64_t offset_b = ptr.offset();
    const uint64_t offset_e = offset_b + ptr.id();

    uint64_t block_offset_b = offset_b / block_size * block_size;
    uint64_t block_offset_e = offset_e;
    for (uint64_t o = block_offset_b; o < block_offset_e; o += block_size) {
      uint8_t* vm_addr = reinterpret_cast<uint8_t*>(o);

      if (cache_.is_cached(cache_key(vm_addr))) {
        cache_block& cb = cache_.ensure_cached(cache_key(vm_addr));

        if (cb.flushing) {
          complete_flush();
        }

        uint64_t offset_in_block_b = (offset_b > o) ? offset_b - o : 0;
        uint64_t offset_in_block_e = (offset_e < o + block_size) ? offset_e - o : block_size;
        sections_remove(cb.dirty_sections, {offset_in_block_b, offset_in_block_e});
      }
    }

    allocator_.remote_deallocate(reinterpret_cast<void*>(ptr.offset()), ptr.id(), ptr.owner());
  }
}

PCAS_TEST_CASE("[pcas::pcas] malloc and free with block policy") {
  pcas pc;
  constexpr int n = 10;
  PCAS_SUBCASE("free immediately") {
    for (int i = 1; i < n; i++) {
      auto p = pc.malloc<int, mem_mapper::block>(i * 1234);
      pc.free(p);
    }
  }
  PCAS_SUBCASE("free after accumulation") {
    pcas::global_ptr<int> ptrs[n];
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
  constexpr int n = 10;
  PCAS_SUBCASE("free immediately") {
    for (int i = 1; i < n; i++) {
      auto p = pc.malloc<int, mem_mapper::cyclic>(i * 123456);
      pc.free(p);
    }
  }
  PCAS_SUBCASE("free after accumulation") {
    pcas::global_ptr<int> ptrs[n];
    for (int i = 1; i < n; i++) {
      ptrs[i] = pc.malloc<int, mem_mapper::cyclic>(i * 27438, pcas::block_size * i);
    }
    for (int i = 1; i < n; i++) {
      pc.free(ptrs[i]);
    }
  }
}

PCAS_TEST_CASE("[pcas::pcas] malloc and free (local)") {
  pcas pc;
  constexpr int n = 10;
  PCAS_SUBCASE("free immediately") {
    for (int i = 0; i < n; i++) {
      auto p = pc.malloc_local<int>((uint64_t)1 << i);
      pc.free(p);
    }
  }
  PCAS_SUBCASE("free after accumulation") {
    pcas::global_ptr<int> ptrs[n];
    for (int i = 0; i < n; i++) {
      ptrs[i] = pc.malloc_local<int>((uint64_t)1 << i);
    }
    for (int i = 0; i < n; i++) {
      pc.free(ptrs[i]);
    }
  }
  PCAS_SUBCASE("remote free") {
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    pcas::global_ptr<int> ptrs_send[n];
    pcas::global_ptr<int> ptrs_recv[n];
    for (int i = 0; i < n; i++) {
      ptrs_send[i] = pc.malloc_local<int>((uint64_t)1 << i);
    }

    MPI_Request req_send, req_recv;
    MPI_Isend(ptrs_send, sizeof(pcas::global_ptr<int>) * n, MPI_BYTE, (nproc + rank + 1) % nproc, 0, MPI_COMM_WORLD, &req_send);
    MPI_Irecv(ptrs_recv, sizeof(pcas::global_ptr<int>) * n, MPI_BYTE, (nproc + rank - 1) % nproc, 0, MPI_COMM_WORLD, &req_recv);
    MPI_Wait(&req_send, MPI_STATUS_IGNORE);
    MPI_Wait(&req_recv, MPI_STATUS_IGNORE);

    for (int i = 0; i < n; i++) {
      pc.free(ptrs_recv[i]);
    }
  }
}

template <typename P>
template <typename T, typename Func>
inline void pcas_if<P>::for_each_block(global_ptr<T> ptr, uint64_t nelems, Func fn) {
  mem_obj& mo = get_mem_obj(ptr.id());

  uint64_t offset_min = ptr.offset();
  uint64_t offset_max = offset_min + nelems * sizeof(T);
  uint64_t offset     = offset_min;

  PCAS_CHECK(offset_max <= mo.size());

  while (offset < offset_max) {
    auto bi = mo.mem_mapper().get_block_info(offset);
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
template <typename ConstT, typename T>
inline std::enable_if_t<std::is_same_v<std::remove_const_t<ConstT>, T>>
pcas_if<P>::get(global_ptr<ConstT> from_ptr, T* to_ptr, uint64_t nelems) {
  uint64_t size = nelems * sizeof(T);
  auto ev = logger::template record<logger_kind::Get>(size);

  T* raw_ptr;
  if (from_ptr.owner() == -1) {
    raw_ptr = (T*)checkout_impl<access_mode::read, false>(static_cast<global_ptr<uint8_t>>(from_ptr), size);
  } else {
    raw_ptr = (T*)checkout_impl_local<access_mode::read, false>(static_cast<global_ptr<uint8_t>>(from_ptr), size);
  }

  std::memcpy(to_ptr, raw_ptr, size);
}

template <typename P>
template <typename T>
inline void pcas_if<P>::put(const T* from_ptr, global_ptr<T> to_ptr, uint64_t nelems) {
  static_assert(!std::is_const_v<T>);

  uint64_t size = nelems * sizeof(T);
  auto ev = logger::template record<logger_kind::Put>(size);

  T* raw_ptr;
  if (to_ptr.owner() == -1) {
    raw_ptr = (T*)checkout_impl<access_mode::write, false>(static_cast<global_ptr<uint8_t>>(to_ptr), size);
  } else {
    raw_ptr = (T*)checkout_impl_local<access_mode::write, false>(static_cast<global_ptr<uint8_t>>(to_ptr), size);
  }

  std::memcpy(raw_ptr, from_ptr, size);

  if (to_ptr.owner() == -1) {
    checkin_impl<false>(static_cast<global_ptr<uint8_t>>(to_ptr), size, access_mode::write);
  } else {
    checkin_impl_local<false>(static_cast<global_ptr<uint8_t>>(to_ptr), size, access_mode::write);
  }
}

template <typename P>
template <typename ConstT, typename T>
inline std::enable_if_t<std::is_same_v<std::remove_const_t<ConstT>, T>>
pcas_if<P>::get_nocache(global_ptr<ConstT> from_ptr, T* to_ptr, uint64_t nelems) {
  uint64_t size = nelems * sizeof(T);
  auto ev = logger::template record<logger_kind::Get>(size);

  if (from_ptr.owner() != -1) {
    die("unimplemented");
  }

  mem_obj& mo = get_mem_obj(from_ptr.id());

  const uint64_t offset_b = from_ptr.offset();
  const uint64_t offset_e = offset_b + nelems * sizeof(T);

  std::vector<MPI_Request> reqs;

  for_each_block(from_ptr, nelems, [&](const auto& bi) {
    uint64_t block_offset_b = std::max(bi.offset_b, offset_b);
    uint64_t block_offset_e = std::min(bi.offset_e, offset_e);
    uint64_t pm_offset = bi.pm_offset + block_offset_b - bi.offset_b;

    if (topo_.is_locally_accessible(bi.owner)) {
      int target_intra_rank = topo_.intra_rank(bi.owner);
      void* from_vm_addr = mo.home_pm(target_intra_rank).anon_vm_addr();
      std::memcpy((uint8_t*)to_ptr - offset_b + block_offset_b,
                  (uint8_t*)from_vm_addr + pm_offset,
                  block_offset_e - block_offset_b);
    } else {
      MPI_Request req;
      MPI_Rget((uint8_t*)to_ptr - offset_b + block_offset_b,
               block_offset_e - block_offset_b,
               MPI_UINT8_T,
               bi.owner,
               pm_offset,
               block_offset_e - block_offset_b,
               MPI_UINT8_T,
               mo.win(),
               &req);
      reqs.push_back(req);
    }
  });

  MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
}

template <typename P>
template <typename T>
inline void pcas_if<P>::put_nocache(const T* from_ptr, global_ptr<T> to_ptr, uint64_t nelems) {
  static_assert(!std::is_const_v<T>);

  uint64_t size = nelems * sizeof(T);
  auto ev = logger::template record<logger_kind::Put>(size);

  if (to_ptr.owner() != -1) {
    die("unimplemented");
  }

  mem_obj& mo = get_mem_obj(to_ptr.id());

  const uint64_t offset_b = to_ptr.offset();
  const uint64_t offset_e = offset_b + nelems * sizeof(T);

  for_each_block(to_ptr, nelems, [&](const auto& bi) {
    uint64_t block_offset_b = std::max(bi.offset_b, offset_b);
    uint64_t block_offset_e = std::min(bi.offset_e, offset_e);
    uint64_t pm_offset = bi.pm_offset + block_offset_b - bi.offset_b;

    if (topo_.is_locally_accessible(bi.owner)) {
      int target_intra_rank = topo_.intra_rank(bi.owner);
      void* to_vm_addr = mo.home_pm(target_intra_rank).anon_vm_addr();
      std::memcpy((uint8_t*)to_vm_addr + pm_offset,
                  (uint8_t*)from_ptr - offset_b + block_offset_b,
                  block_offset_e - block_offset_b);
    } else {
      MPI_Put((uint8_t*)from_ptr - offset_b + block_offset_b,
              block_offset_e - block_offset_b,
              MPI_UINT8_T,
              bi.owner,
              pm_offset,
              block_offset_e - block_offset_b,
              MPI_UINT8_T,
              mo.win());
    }
  });

  // ensure remote completion
  MPI_Win_flush_all(mo.win());
}

PCAS_TEST_CASE("[pcas::pcas] get and put") {
  pcas pc;

  int rank = pc.rank();

  int n = 1000000;

  pcas::global_ptr<int> ps[2];
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

PCAS_TEST_CASE("[pcas::pcas] get and put (nocache)") {
  pcas pc;

  int rank = pc.rank();

  int n = 1000000;

  pcas::global_ptr<int> ps[2];
  ps[0] = pc.malloc<int, mem_mapper::block >(n);
  ps[1] = pc.malloc<int, mem_mapper::cyclic>(n);

  int* buf = new int[n + 2];

  for (auto p : ps) {
    if (rank == 0) {
      for (int i = 0; i < n; i++) {
        buf[i] = i;
      }
      pc.put_nocache(buf, p, n);
    }

    pc.barrier();

    PCAS_SUBCASE("get the entire array") {
      int special = 417;
      buf[0] = buf[n + 1] = special;

      pc.get_nocache(p, buf + 1, n);

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

      pc.get_nocache(p + ib, buf + 1, s);

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
        pc.get_nocache(p + i, &buf[1], 1);
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

  mem_obj& mo = get_mem_obj(ptr.id());

  const uint64_t offset_b = ptr.offset();
  const uint64_t offset_e = offset_b + size;

  try {
    for_each_block(ptr, size, [&](const auto& bi) {
      if (!mo.is_locally_accessible(bi.owner)) {
        uint64_t block_offset_b = std::max(bi.offset_b, offset_b / block_size * block_size);
        uint64_t block_offset_e = std::min(bi.offset_e, offset_e);
        for (uint64_t o = block_offset_b; o < block_offset_e; o += block_size) {
          uint8_t* vm_addr = reinterpret_cast<uint8_t*>(mo.vm().addr()) + o;
          cache_block& cb = cache_.ensure_cached(cache_key(vm_addr));

          if (!cb.mapped) {
            cb.vm_addr = vm_addr;
            cb.owner = bi.owner;
            cb.pm_offset = bi.pm_offset + o - bi.offset_b;
            cb.win = mo.win();
            remap_cache_blocks_.push_back(&cb);

            PCAS_CHECK(cb.pm_offset + block_size <= mo.mem_mapper().get_local_size(cb.owner));
          }

          cb.transaction_id = transaction_id_;

          fetch_begin(cb);
        }
      }
    });
  } catch (cache_full_exception& e) {
    // do not go further
  }

  transaction_id_++;
}

template <typename P>
template <access_mode Mode, typename T>
inline std::conditional_t<Mode == access_mode::read, const T*, T*>
pcas_if<P>::checkout(global_ptr<T> ptr, uint64_t nelems) {
  // If T is const, then it cannot be checked out with write access mode
  static_assert(!std::is_const_v<T> || Mode == access_mode::read);

  uint64_t size = nelems * sizeof(T);
  auto ev = logger::template record<logger_kind::Checkout>(size);

  void* raw_ptr;
  if (ptr.owner() == -1) {
    raw_ptr = checkout_impl<Mode, true>(static_cast<global_ptr<uint8_t>>(ptr), size);
  } else {
    raw_ptr = checkout_impl_local<Mode, true>(static_cast<global_ptr<uint8_t>>(ptr), size);
  }

  auto c = find_checkout_entry(raw_ptr, size);
  if (c) {
    // Only read-only access is allowed for overlapped checkout
    // TODO: dynamic check for conflicting access when the region is overlapped but not the same
    PCAS_CHECK((*c)->mode == access_mode::read);
    PCAS_CHECK(Mode == access_mode::read);
    (*c)->count++;
  } else {
    add_checkout_entry({
      .ptr = static_cast<global_ptr<uint8_t>>(ptr), .raw_ptr = raw_ptr,
      .size = size, .mode = Mode, .count = 1,
    });
  }

  return reinterpret_cast<std::remove_const_t<T>*>(raw_ptr);
}

template <typename P>
template <access_mode Mode, bool DoCheckout>
inline void* pcas_if<P>::checkout_impl(global_ptr<uint8_t> ptr, uint64_t size) {
  mem_obj& mo = get_mem_obj(ptr.id());

  const uint64_t offset_b = ptr.offset();
  const uint64_t offset_e = offset_b + size;

  PCAS_CHECK(offset_e <= mo.size());

  uint8_t* raw_ptr = reinterpret_cast<uint8_t*>(mo.vm().addr()) + offset_b;

  // fast path for small requests using TLB
  {
    std::optional<home_block*> hbe = home_tlb_.find([&](const home_block* hb) {
      return hb->vm_addr <= raw_ptr && raw_ptr + size <= hb->vm_addr + hb->size;
    });
    if (hbe.has_value()) {
      home_block& hb = **hbe;
      PCAS_ASSERT(hb.mapped);
      if (DoCheckout) {
        hb.checkout_count++;
      }
      return raw_ptr;
    }

    bool is_within_block = offset_b / block_size == (offset_e - 1) / block_size;
    if (is_within_block) { // TLB works only for requests within each cache block
      std::optional<cache_block*> cbe = cache_tlb_.find([&](const cache_block* cb) {
        return cb->vm_addr <= raw_ptr && raw_ptr + size <= cb->vm_addr + block_size;
      });
      if (cbe.has_value()) {
        cache_block& cb = **cbe;
        PCAS_ASSERT(cb.cstate == cache_state::valid);
        PCAS_ASSERT(cb.mapped);

        if (cb.flushing && Mode != access_mode::read) {
          auto ev2 = logger::template record<logger_kind::FlushConflicted>();
          complete_flush();
        }

        if (Mode == access_mode::write) {
          uint64_t offset_in_block_b = raw_ptr - cb.vm_addr;
          uint64_t offset_in_block_e = raw_ptr + size - cb.vm_addr;
          sections_insert(cb.partial_sections, {offset_in_block_b, offset_in_block_e});
        }

        if (Mode != access_mode::write &&
            *cb.partial_sections.begin() != section{0, block_size}) {
          // fetch the rest of the block
          fetch_begin(cb);

          // Immediately wait for communication completion here because it is a fast path;
          // i.e., there is only one cache block to be checked out.
          PCAS_CHECK(cb.req != MPI_REQUEST_NULL);
          MPI_Wait(&cb.req, MPI_STATUS_IGNORE);
          PCAS_CHECK(cb.req == MPI_REQUEST_NULL);

          cb.cstate = cache_state::valid;
        }

        if (DoCheckout) {
          cb.checkout_count++;
        }
        return raw_ptr;
      }
    }
  }

  uint64_t size_pf = size;
  if (Mode != access_mode::write) {
    size_pf = mo.size_with_prefetch(offset_b, size);
  }

  std::vector<MPI_Request> reqs;

  for_each_block(ptr, size_pf, [&](const auto& bi) {
    if (topo_.is_locally_accessible(bi.owner)) {
      bool is_prefetch = bi.offset_b >= offset_e;
      if (!is_prefetch) {
        uint8_t* vm_addr = reinterpret_cast<uint8_t*>(mo.vm().addr()) + bi.offset_b;

        home_block& hb = ensure_mmap_cached_strict(vm_addr);

        if (!hb.mapped) {
          int target_intra_rank = topo_.intra_rank(bi.owner);
          hb.pm = &mo.home_pm(target_intra_rank);
          hb.pm_offset = bi.pm_offset;
          hb.vm_addr = vm_addr;
          hb.size = bi.offset_e - bi.offset_b;
          remap_home_blocks_.push_back(&hb);
        }

        hb.transaction_id = transaction_id_;

        if (DoCheckout) {
          hb.checkout_count++;
        }

        home_tlb_.add(&hb);
      }
    } else {
      // for each cache block
      uint64_t block_offset_b = std::max(bi.offset_b, offset_b / block_size * block_size);
      uint64_t block_offset_e = std::min(bi.offset_e, offset_b + size_pf);
      for (uint64_t o = block_offset_b; o < block_offset_e; o += block_size) {
        uint8_t* vm_addr = reinterpret_cast<uint8_t*>(mo.vm().addr()) + o;

        cache_block& cb = ensure_cached_strict(vm_addr);

        if (!cb.mapped) {
          cb.vm_addr = vm_addr;
          cb.owner = bi.owner;
          cb.pm_offset = bi.pm_offset + o - bi.offset_b;
          cb.win = mo.win();
          remap_cache_blocks_.push_back(&cb);

          PCAS_CHECK(cb.pm_offset + block_size <= mo.mem_mapper().get_local_size(cb.owner));
        }

        cb.transaction_id = transaction_id_;

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
          uint64_t offset_in_block_b = (offset_b > o) ? offset_b - o : 0;
          uint64_t offset_in_block_e = (offset_e < o + block_size) ? offset_e - o : block_size;
          sections_insert(cb.partial_sections, {offset_in_block_b, offset_in_block_e});
        }

        // Suppose that a cache block is represented as [a1, a2].
        // If a1 is checked out with write-only access mode, then [a1, a2] is allocated a cache entry,
        // but fetch for a1 and a2 is skipped.  Later, if a2 is checked out with read access mode,
        // the data for a2 would not be fetched because it is already in the cache.
        // Thus, we allocate a `partial` flag to each cache entry to indicate if the entire cache block
        // has been already fetched or not.
        if (Mode != access_mode::write) {
          fetch_begin(cb);
        }

        bool is_prefetch = o >= offset_e;
        if (!is_prefetch) {
          if (DoCheckout) {
            cb.checkout_count++;
          }

          if (cb.cstate == cache_state::fetching) {
            PCAS_CHECK(cb.req != MPI_REQUEST_NULL);
            reqs.push_back(cb.req);
            cb.req = MPI_REQUEST_NULL;
          }

          cb.cstate = cache_state::valid;

          cache_tlb_.add(&cb);
        }
      }
    }
  });

  // If the transaction ID of a cache entry is equal to the current transaction ID (outer.transaction_id_),
  // we do not evict it from the cache.
  transaction_id_++;

  // Overlap communication and memory remapping
  ensure_remapped();

  if (!reqs.empty()) {
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
  }

  return raw_ptr;
}

template <typename P>
template <access_mode Mode, bool DoCheckout>
inline void* pcas_if<P>::checkout_impl_local(global_ptr<uint8_t> ptr, uint64_t size) {
  const int target_rank = ptr.owner();
  PCAS_CHECK(0 <= target_rank);
  PCAS_CHECK(target_rank < nproc());

  if (topo_.is_locally_accessible(target_rank)) {
    return reinterpret_cast<void*>(ptr.offset());
  }

  const uint64_t offset_b = ptr.offset();
  const uint64_t offset_e = offset_b + size;

  std::vector<MPI_Request> reqs;

  uint64_t block_offset_b = offset_b / block_size * block_size;
  uint64_t block_offset_e = offset_e;
  for (uint64_t o = block_offset_b; o < block_offset_e; o += block_size) {
    uint8_t* vm_addr = reinterpret_cast<uint8_t*>(o);

    cache_block& cb = ensure_cached_strict(vm_addr);

    if (!cb.mapped) {
      cb.vm_addr = vm_addr;
      cb.owner = target_rank;
      cb.pm_offset = reinterpret_cast<uint64_t>(vm_addr);
      cb.win = allocator_.win();
      remap_cache_blocks_.push_back(&cb);
    }

    cb.transaction_id = transaction_id_;

    if (cb.flushing && Mode != access_mode::read) {
      auto ev2 = logger::template record<logger_kind::FlushConflicted>();
      complete_flush();
    }

    if (Mode == access_mode::write) {
      uint64_t offset_in_block_b = (offset_b > o) ? offset_b - o : 0;
      uint64_t offset_in_block_e = (offset_e < o + block_size) ? offset_e - o : block_size;
      sections_insert(cb.partial_sections, {offset_in_block_b, offset_in_block_e});
    }

    if (Mode != access_mode::write) {
      fetch_begin(cb);
    }

    if (DoCheckout) {
      cb.checkout_count++;
    }

    if (cb.cstate == cache_state::fetching) {
      PCAS_CHECK(cb.req != MPI_REQUEST_NULL);
      reqs.push_back(cb.req);
      cb.req = MPI_REQUEST_NULL;
    }

    cb.cstate = cache_state::valid;
  }

  transaction_id_++;

  ensure_remapped();

  if (!reqs.empty()) {
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
  }

  return reinterpret_cast<void*>(offset_b);
}

template <typename P>
template <typename T>
inline void pcas_if<P>::checkin(T* raw_ptr, uint64_t nelems) {
  // TODO: judge access mode by its type (const/nonconst)?

  uint64_t size = nelems * sizeof(T);
  auto ev = logger::template record<logger_kind::Checkin>();

  auto c = find_checkout_entry((void*)raw_ptr, size);
  if (!c) {
    die("The region [%p, %p) passed to checkin() is not registered", raw_ptr, raw_ptr + size);
  }

  global_ptr<uint8_t> ptr = (*c)->ptr;
  if (ptr.owner() == -1) {
    checkin_impl<true>(ptr, size, (*c)->mode);
  } else {
    checkin_impl_local<true>(ptr, size, (*c)->mode);
  }

  if (--(*c)->count == 0) {
    remove_checkout_entry(c);
  }
}

template <typename P>
template <bool DoCheckout>
inline void pcas_if<P>::checkin_impl(global_ptr<uint8_t> ptr, uint64_t size, access_mode mode) {
  mem_obj& mo = get_mem_obj(ptr.id());

  const uint64_t offset_b = ptr.offset();
  const uint64_t offset_e = offset_b + size;

  uint8_t* raw_ptr = reinterpret_cast<uint8_t*>(mo.vm().addr()) + offset_b;

  // fast path for small requests using TLB
  {
    std::optional<home_block*> hbe = home_tlb_.find([&](const home_block* hb) {
      return hb->vm_addr <= raw_ptr && raw_ptr + size <= hb->vm_addr + hb->size;
    });
    if (hbe.has_value()) {
      home_block& hb = **hbe;
      PCAS_ASSERT(hb.mapped);
      if (DoCheckout) {
        hb.checkout_count--;
      }
      return;
    }

    bool is_within_block = offset_b / block_size == (offset_e - 1) / block_size;
    if (is_within_block) {
      std::optional<cache_block*> cbe = cache_tlb_.find([&](const cache_block* cb) {
        return cb->vm_addr <= raw_ptr && raw_ptr + size <= cb->vm_addr + block_size;
      });
      if (cbe.has_value()) {
        cache_block& cb = **cbe;
        PCAS_ASSERT(cb.cstate == cache_state::valid);
        PCAS_ASSERT(cb.mapped);

        if (mode != access_mode::read) {
          bool is_new_dirty_block = cb.dirty_sections.empty();

          uint64_t offset_in_block_b = raw_ptr - cb.vm_addr;
          uint64_t offset_in_block_e = raw_ptr + size - cb.vm_addr;
          sections_insert(cb.dirty_sections, {offset_in_block_b, offset_in_block_e});

          if (is_new_dirty_block) {
            add_dirty_cache_block(cb);
          }
        }

        if (DoCheckout) {
          cb.checkout_count--;
        }
        return;
      }
    }
  }

  for_each_block(ptr, size, [&](const auto& bi) {
    if (topo_.is_locally_accessible(bi.owner)) {
      if (DoCheckout) {
        uint8_t* vm_addr = reinterpret_cast<uint8_t*>(mo.vm().addr()) + bi.offset_b;
        home_block& hb = mmap_cache_.template ensure_cached<false>(cache_key(vm_addr));
        hb.checkout_count--;
      }
    } else {
      uint64_t block_offset_b = std::max(bi.offset_b, offset_b / block_size * block_size);
      uint64_t block_offset_e = std::min(bi.offset_e, offset_e);
      for (uint64_t o = block_offset_b; o < block_offset_e; o += block_size) {
        uint8_t* vm_addr = reinterpret_cast<uint8_t*>(mo.vm().addr()) + o;
        cache_block& cb = cache_.template ensure_cached<false>(cache_key(vm_addr));

        if (mode != access_mode::read) {
          bool is_new_dirty_block = cb.dirty_sections.empty();

          uint64_t offset_in_block_b = (offset_b > o) ? offset_b - o : 0;
          uint64_t offset_in_block_e = (offset_e < o + block_size) ? offset_e - o : block_size;
          sections_insert(cb.dirty_sections, {offset_in_block_b, offset_in_block_e});

          if (is_new_dirty_block) {
            add_dirty_cache_block(cb);
          }
        }

        if (DoCheckout) {
          cb.checkout_count--;
        }
      }
    }
  });
}

template <typename P>
template <bool DoCheckout>
inline void pcas_if<P>::checkin_impl_local(global_ptr<uint8_t> ptr, uint64_t size, access_mode mode) {
  const int target_rank = ptr.owner();
  PCAS_CHECK(0 <= target_rank);
  PCAS_CHECK(target_rank < nproc());

  if (topo_.is_locally_accessible(target_rank)) {
    return;
  }

  const uint64_t offset_b = ptr.offset();
  const uint64_t offset_e = offset_b + size;

  uint64_t block_offset_b = offset_b / block_size * block_size;
  uint64_t block_offset_e = offset_e;
  for (uint64_t o = block_offset_b; o < block_offset_e; o += block_size) {
    uint8_t* vm_addr = reinterpret_cast<uint8_t*>(o);
    cache_block& cb = cache_.template ensure_cached<false>(cache_key(vm_addr));

    if (mode != access_mode::read) {
      bool is_new_dirty_block = cb.dirty_sections.empty();

      uint64_t offset_in_block_b = (offset_b > o) ? offset_b - o : 0;
      uint64_t offset_in_block_e = (offset_e < o + block_size) ? offset_e - o : block_size;
      sections_insert(cb.dirty_sections, {offset_in_block_b, offset_in_block_e});

      if (is_new_dirty_block) {
        add_dirty_cache_block(cb);
      }
    }

    if (DoCheckout) {
      cb.checkout_count--;
    }
  }
}

PCAS_TEST_CASE("[pcas::pcas] checkout and checkin (small, aligned)") {
  pcas pc;

  int rank = pc.rank();
  int nproc = pc.nproc();

  int n = pcas::block_size * nproc;
  pcas::global_ptr<uint8_t> ps[2];
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

  pcas::global_ptr<int> ps[2];
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

PCAS_TEST_CASE("[pcas::pcas] checkout and checkin (local)") {
  pcas pc(16 * pcas::block_size);

  int rank = pc.rank();
  int nproc = pc.nproc();

  PCAS_SUBCASE("list creation") {
    int niter = 10000;
    int n_alloc_iter = 100;

    struct node_t {
      pcas::global_ptr<node_t> next;
      int value;
    };

    pcas::global_ptr<node_t> root_node = pc.malloc_local<node_t>(1);

    {
      node_t* n = pc.checkout<access_mode::write>(root_node, 1);
      n->next = pcas::global_ptr<node_t>{};
      n->value = rank;
      pc.checkin(n, 1);
    }

    pcas::global_ptr<node_t> node = root_node;
    for (int i = 0; i < niter; i++) {
      for (int j = 0; j < n_alloc_iter; j++) {
        // append a new node
        pcas::global_ptr<node_t> new_node = pc.malloc_local<node_t>(1);

        {
          pcas::global_ptr<node_t>* next = pc.checkout<access_mode::write>(&(node->*(&node_t::next)), 1);
          *next = new_node;
          pc.checkin(next, 1);
        }

        int val = std::invoke([&]() {
          const node_t* n = pc.checkout<access_mode::read>(node, 1);
          int val = n->value;
          pc.checkin(n, 1);
          return val;
        });

        {
          node_t* n = pc.checkout<access_mode::write>(new_node, 1);
          n->next = pcas::global_ptr<node_t>{};
          n->value = val + 1;
          pc.checkin(n, 1);
        }

        node = new_node;
      }

      pc.release();

      // exchange nodes across nodes
      pcas::global_ptr<node_t> next_node;

      MPI_Request req_send, req_recv;
      MPI_Isend(&node     , sizeof(pcas::global_ptr<node_t>), MPI_BYTE, (nproc + rank + 1) % nproc, i, MPI_COMM_WORLD, &req_send);
      MPI_Irecv(&next_node, sizeof(pcas::global_ptr<node_t>), MPI_BYTE, (nproc + rank - 1) % nproc, i, MPI_COMM_WORLD, &req_recv);
      MPI_Wait(&req_send, MPI_STATUS_IGNORE);
      MPI_Wait(&req_recv, MPI_STATUS_IGNORE);

      node = next_node;

      pc.acquire();
    }

    pc.barrier();

    int count = 0;
    node = root_node;
    while (node != pcas::global_ptr<node_t>{}) {
      const node_t* n = pc.checkout<access_mode::read>(node, 1);

      PCAS_CHECK(n->value == rank + count);

      pcas::global_ptr<node_t> prev_node = node;
      node = n->next;

      pc.checkin(n, 1);

      pc.free(prev_node);

      count++;
    }
  }
}

// TODO: add tests to below functions

template <typename P>
inline void pcas_if<P>::release() {
  auto ev = logger::template record<logger_kind::Release>();
  ensure_all_cache_clean();
}

template <typename P>
inline void pcas_if<P>::release_lazy(release_handler* handler) {
  PCAS_CHECK(empty(checkouts_));

  epoch_t next_epoch = cache_dirty_ ? rm_.epoch() + 1 : 0; // 0 means clean
  *handler = {.rank = rank(), .epoch = next_epoch};
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
  MPI_Barrier(topo_.global_comm());
  acquire();
}

template <typename P>
inline void pcas_if<P>::poll() {
  if (rm_.request() > rm_.epoch()) {
    auto ev = logger::template record<logger_kind::ReleaseLazy>();
    PCAS_CHECK(rm_.request() == rm_.epoch() + 1);

    ensure_all_cache_clean();

    PCAS_CHECK(rm_.request() == rm_.epoch());
  }
}

}
