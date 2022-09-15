#pragma once

#include "pcas/util.hpp"

namespace pcas {
namespace mem_mapper {

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

class base {
protected:
  uint64_t size_;
  int nproc_;

public:
  base(uint64_t size, int nproc) : size_(size), nproc_(nproc) {}

  virtual ~base() = default;

  virtual uint64_t get_local_size(int rank) = 0;

  virtual uint64_t get_effective_size() = 0;

  // Returns the block info that specifies the owner and the range [offset_b, offset_e)
  // of the block that contains the given offset.
  // pm_offset is an offset of the owner's local physical memory, corresponding to the given offset.
  virtual block_info get_block_info(uint64_t offset) = 0;
};

class block : public base {
public:
  using base::base;

  uint64_t get_local_size(int rank [[maybe_unused]]) {
    uint64_t nblock_g = (size_ + min_block_size - 1) / min_block_size;
    uint64_t nblock_l = (nblock_g + nproc_ - 1) / nproc_;
    return nblock_l * min_block_size;
  }

  uint64_t get_effective_size() {
    return get_local_size(0) * nproc_;
  }

  block_info get_block_info(uint64_t offset) {
    PCAS_CHECK(offset < get_effective_size());
    uint64_t size_l    = get_local_size(0);
    int      owner     = offset / size_l;
    uint64_t offset_b  = owner * size_l;
    uint64_t offset_e  = std::min((owner + 1) * size_l, size_);
    uint64_t pm_offset = offset - size_l * owner;
    return block_info{owner, offset_b, offset_e, pm_offset};
  }
};

PCAS_TEST_CASE("[pcas::mem_mapper::block] calculate local block size") {
  auto local_block_size = [](uint64_t size, int nproc) -> uint64_t {
    return block(size, nproc).get_local_size(0);
  };
  PCAS_CHECK(local_block_size(min_block_size * 4     , 4) == min_block_size    );
  PCAS_CHECK(local_block_size(min_block_size * 12    , 4) == min_block_size * 3);
  PCAS_CHECK(local_block_size(min_block_size * 13    , 4) == min_block_size * 4);
  PCAS_CHECK(local_block_size(min_block_size * 12 + 1, 4) == min_block_size * 4);
  PCAS_CHECK(local_block_size(min_block_size * 12 - 1, 4) == min_block_size * 3);
  PCAS_CHECK(local_block_size(1                      , 4) == min_block_size    );
  PCAS_CHECK(local_block_size(1                      , 1) == min_block_size    );
  PCAS_CHECK(local_block_size(min_block_size * 3     , 1) == min_block_size * 3);
}

PCAS_TEST_CASE("[pcas::mem_mapper::block] get block information at specified offset") {
  auto block_index_info = [](uint64_t offset, uint64_t size, int nproc) -> block_info {
    return block(size, nproc).get_block_info(offset);
  };
  uint64_t mb = min_block_size;
  PCAS_CHECK(block_index_info(0         , mb * 4     , 4) == (block_info{0, 0     , mb         , 0     }));
  PCAS_CHECK(block_index_info(mb        , mb * 4     , 4) == (block_info{1, mb    , mb * 2     , 0     }));
  PCAS_CHECK(block_index_info(mb * 2    , mb * 4     , 4) == (block_info{2, mb * 2, mb * 3     , 0     }));
  PCAS_CHECK(block_index_info(mb * 3    , mb * 4     , 4) == (block_info{3, mb * 3, mb * 4     , 0     }));
  PCAS_CHECK(block_index_info(mb * 4 - 1, mb * 4     , 4) == (block_info{3, mb * 3, mb * 4     , mb - 1}));
  PCAS_CHECK(block_index_info(0         , mb * 12    , 4) == (block_info{0, 0     , mb * 3     , 0     }));
  PCAS_CHECK(block_index_info(mb        , mb * 12    , 4) == (block_info{0, 0     , mb * 3     , mb    }));
  PCAS_CHECK(block_index_info(mb * 3    , mb * 12    , 4) == (block_info{1, mb * 3, mb * 6     , 0     }));
  PCAS_CHECK(block_index_info(mb * 11   , mb * 12 - 1, 4) == (block_info{3, mb * 9, mb * 12 - 1, mb * 2}));
}

}
}
