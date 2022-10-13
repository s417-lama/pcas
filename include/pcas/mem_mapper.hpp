#pragma once

#include "pcas/util.hpp"
#include "pcas/global_ptr.hpp"

namespace pcas {

struct block_info {
  int      owner;
  uint64_t offset_b;
  uint64_t offset_e;
  uint64_t pm_offset;

  bool operator==(const block_info& b) const {
    return owner == b.owner && offset_b == b.offset_b && offset_e == b.offset_e && pm_offset == b.pm_offset;
  }
  bool operator!=(const block_info& b) const {
    return !(*this == b);
  }
};

enum class access_mode {
  read,
  write,
  read_write,
};

struct mem_obj;

class mem_mapper_base {
public:
  virtual ~mem_mapper_base() = default;

  // TODO: make them nonvirtual
  virtual uint64_t get_local_size(int rank) = 0;
  virtual uint64_t get_effective_size() = 0;
  virtual block_info get_block_info(uint64_t offset) = 0;

  virtual void* checkout_impl(void*               this_p,
                              global_ptr<uint8_t> ptr,
                              uint64_t            size,
                              access_mode         mode) = 0;
};

// CRTP
template <template<typename> typename Derived, typename P>
class mem_mapper : public mem_mapper_base {
  using derived_t = Derived<P>;

protected:
  uint64_t size_;
  int nproc_;

public:
  mem_mapper(uint64_t size, int nproc) : size_(size), nproc_(nproc) {}

  void* checkout_impl(void*               this_p,
                      global_ptr<uint8_t> ptr,
                      uint64_t            size,
                      access_mode         mode) override {
    return ((P*)this_p)->template checkout_impl<derived_t>(ptr, size, mode);
  }

  template <typename Fn>
  inline void for_each_block(uint64_t offset_b, uint64_t offset_e, Fn&& fn) {
    uint64_t offset = offset_b;
    while (offset < offset_e) {
      const block_info bi = static_cast<derived_t&>(*this)._get_block_info(offset);
      std::forward<Fn>(fn)(bi);
      offset = bi.offset_e;
    }
  }

};

// Preset block distribution
// -----------------------------------------------------------------------------

template <typename P>
class mem_mapper_block : public mem_mapper<mem_mapper_block, P> {
public:
  using mem_mapper<mem_mapper_block, P>::mem_mapper;

  uint64_t get_local_size(int rank [[maybe_unused]]) {
    uint64_t nblock_g = (this->size_ + P::block_size - 1) / P::block_size;
    uint64_t nblock_l = (nblock_g + this->nproc_ - 1) / this->nproc_;
    return nblock_l * P::block_size;
  }

  uint64_t get_effective_size() {
    return get_local_size(0) * this->nproc_;
  }

  block_info _get_block_info(uint64_t offset) {
    PCAS_CHECK(offset < get_effective_size());
    uint64_t size_l    = get_local_size(0);
    int      owner     = offset / size_l;
    uint64_t offset_b  = owner * size_l;
    uint64_t offset_e  = std::min((owner + 1) * size_l, this->size_);
    return block_info{owner, offset_b, offset_e, 0};
  }
  block_info get_block_info(uint64_t offset) {
    PCAS_CHECK(offset < get_effective_size());
    uint64_t size_l    = get_local_size(0);
    int      owner     = offset / size_l;
    uint64_t offset_b  = owner * size_l;
    uint64_t offset_e  = std::min((owner + 1) * size_l, this->size_);
    return block_info{owner, offset_b, offset_e, 0};
  }
};

struct test_policy {
  static constexpr uint64_t block_size = 65536;
  template <typename MemMapper>
  void* checkout_impl(global_ptr<uint8_t>, uint64_t, access_mode) { return nullptr; }
};

PCAS_TEST_CASE("[pcas::mem_mapper_block] calculate local block size") {
  constexpr uint64_t bs = test_policy::block_size;
  auto local_block_size = [](uint64_t size, int nproc) -> uint64_t {
    return mem_mapper_block<test_policy>(size, nproc).get_local_size(0);
  };
  PCAS_CHECK(local_block_size(bs * 4     , 4) == bs    );
  PCAS_CHECK(local_block_size(bs * 12    , 4) == bs * 3);
  PCAS_CHECK(local_block_size(bs * 13    , 4) == bs * 4);
  PCAS_CHECK(local_block_size(bs * 12 + 1, 4) == bs * 4);
  PCAS_CHECK(local_block_size(bs * 12 - 1, 4) == bs * 3);
  PCAS_CHECK(local_block_size(1          , 4) == bs    );
  PCAS_CHECK(local_block_size(1          , 1) == bs    );
  PCAS_CHECK(local_block_size(bs * 3     , 1) == bs * 3);
}

PCAS_TEST_CASE("[pcas::mem_mapper_block] get block information at specified offset") {
  constexpr uint64_t bs = test_policy::block_size;
  auto block_index_info = [](uint64_t offset, uint64_t size, int nproc) -> block_info {
    return mem_mapper_block<test_policy>(size, nproc).get_block_info(offset);
  };
  PCAS_CHECK(block_index_info(0         , bs * 4     , 4) == (block_info{0, 0     , bs         , 0}));
  PCAS_CHECK(block_index_info(bs        , bs * 4     , 4) == (block_info{1, bs    , bs * 2     , 0}));
  PCAS_CHECK(block_index_info(bs * 2    , bs * 4     , 4) == (block_info{2, bs * 2, bs * 3     , 0}));
  PCAS_CHECK(block_index_info(bs * 3    , bs * 4     , 4) == (block_info{3, bs * 3, bs * 4     , 0}));
  PCAS_CHECK(block_index_info(bs * 4 - 1, bs * 4     , 4) == (block_info{3, bs * 3, bs * 4     , 0}));
  PCAS_CHECK(block_index_info(0         , bs * 12    , 4) == (block_info{0, 0     , bs * 3     , 0}));
  PCAS_CHECK(block_index_info(bs        , bs * 12    , 4) == (block_info{0, 0     , bs * 3     , 0}));
  PCAS_CHECK(block_index_info(bs * 3    , bs * 12    , 4) == (block_info{1, bs * 3, bs * 6     , 0}));
  PCAS_CHECK(block_index_info(bs * 11   , bs * 12 - 1, 4) == (block_info{3, bs * 9, bs * 12 - 1, 0}));
}

// Preset cycle distribution
// -----------------------------------------------------------------------------

template <typename P>
class mem_mapper_cyclic : public mem_mapper<mem_mapper_cyclic, P> {
  size_t block_size_;

public:
  mem_mapper_cyclic(uint64_t size, int nproc, size_t block_size = P::block_size)
    : mem_mapper<mem_mapper_cyclic, P>(size, nproc), block_size_(block_size) {
    PCAS_CHECK(block_size >= P::block_size);
    PCAS_CHECK(block_size % P::block_size == 0);
  }

  uint64_t get_local_size(int rank [[maybe_unused]]) {
    uint64_t nblock_g = (this->size_ + block_size_ - 1) / block_size_;
    uint64_t nblock_l = (nblock_g + this->nproc_ - 1) / this->nproc_;
    return nblock_l * block_size_;
  }

  uint64_t get_effective_size() {
    return get_local_size(0) * this->nproc_;
  }

  block_info _get_block_info(uint64_t offset) {
    PCAS_CHECK(offset < get_effective_size());
    uint64_t block_num_g = offset / block_size_;
    uint64_t block_num_l = block_num_g / this->nproc_;
    int      owner       = block_num_g % this->nproc_;
    uint64_t offset_b    = block_num_g * block_size_;
    uint64_t offset_e    = std::min((block_num_g + 1) * block_size_, this->size_);
    uint64_t pm_offset   = block_num_l * block_size_;
    return block_info{owner, offset_b, offset_e, pm_offset};
  }
  block_info get_block_info(uint64_t offset) {
    PCAS_CHECK(offset < get_effective_size());
    uint64_t block_num_g = offset / block_size_;
    uint64_t block_num_l = block_num_g / this->nproc_;
    int      owner       = block_num_g % this->nproc_;
    uint64_t offset_b    = block_num_g * block_size_;
    uint64_t offset_e    = std::min((block_num_g + 1) * block_size_, this->size_);
    uint64_t pm_offset   = block_num_l * block_size_;
    return block_info{owner, offset_b, offset_e, pm_offset};
  }
};

PCAS_TEST_CASE("[pcas::mem_mapper_cyclic] calculate local block size") {
  constexpr uint64_t mb = test_policy::block_size;
  uint64_t bs = mb * 2;
  auto local_block_size = [=](uint64_t size, int nproc) -> uint64_t {
    return mem_mapper_cyclic<test_policy>(size, nproc, bs).get_local_size(0);
  };
  PCAS_CHECK(local_block_size(bs * 4     , 4) == bs    );
  PCAS_CHECK(local_block_size(bs * 12    , 4) == bs * 3);
  PCAS_CHECK(local_block_size(bs * 13    , 4) == bs * 4);
  PCAS_CHECK(local_block_size(bs * 12 + 1, 4) == bs * 4);
  PCAS_CHECK(local_block_size(bs * 12 - 1, 4) == bs * 3);
  PCAS_CHECK(local_block_size(1          , 4) == bs    );
  PCAS_CHECK(local_block_size(1          , 1) == bs    );
  PCAS_CHECK(local_block_size(bs * 3     , 1) == bs * 3);
}

PCAS_TEST_CASE("[pcas::mem_mapper_cyclic] get block information at specified offset") {
  constexpr uint64_t mb = test_policy::block_size;
  uint64_t bs = mb * 2;
  auto block_index_info = [=](uint64_t offset, uint64_t size, int nproc) -> block_info {
    return mem_mapper_cyclic<test_policy>(size, nproc, bs).get_block_info(offset);
  };
  PCAS_CHECK(block_index_info(0         , bs * 4     , 4) == (block_info{0, 0      , bs         , 0     }));
  PCAS_CHECK(block_index_info(bs        , bs * 4     , 4) == (block_info{1, bs     , bs * 2     , 0     }));
  PCAS_CHECK(block_index_info(bs * 2    , bs * 4     , 4) == (block_info{2, bs * 2 , bs * 3     , 0     }));
  PCAS_CHECK(block_index_info(bs * 3    , bs * 4     , 4) == (block_info{3, bs * 3 , bs * 4     , 0     }));
  PCAS_CHECK(block_index_info(bs * 4 - 1, bs * 4     , 4) == (block_info{3, bs * 3 , bs * 4     , 0     }));
  PCAS_CHECK(block_index_info(0         , bs * 12    , 4) == (block_info{0, 0      , bs         , 0     }));
  PCAS_CHECK(block_index_info(bs        , bs * 12    , 4) == (block_info{1, bs     , bs * 2     , 0     }));
  PCAS_CHECK(block_index_info(bs * 3    , bs * 12    , 4) == (block_info{3, bs * 3 , bs * 4     , 0     }));
  PCAS_CHECK(block_index_info(bs * 5 + 2, bs * 12    , 4) == (block_info{1, bs * 5 , bs * 6     , bs    }));
  PCAS_CHECK(block_index_info(bs * 11   , bs * 12 - 1, 4) == (block_info{3, bs * 11, bs * 12 - 1, bs * 2}));
}

}
