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

template <uint64_t BlockSize>
class block : public base {
public:
  using base::base;

  uint64_t get_local_size(int rank [[maybe_unused]]) {
    uint64_t nblock_g = (size_ + BlockSize - 1) / BlockSize;
    uint64_t nblock_l = (nblock_g + nproc_ - 1) / nproc_;
    return nblock_l * BlockSize;
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
  constexpr uint64_t bs = 65536;
  auto local_block_size = [](uint64_t size, int nproc) -> uint64_t {
    return block<bs>(size, nproc).get_local_size(0);
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

PCAS_TEST_CASE("[pcas::mem_mapper::block] get block information at specified offset") {
  constexpr uint64_t bs = 65536;
  auto block_index_info = [](uint64_t offset, uint64_t size, int nproc) -> block_info {
    return block<bs>(size, nproc).get_block_info(offset);
  };
  PCAS_CHECK(block_index_info(0         , bs * 4     , 4) == (block_info{0, 0     , bs         , 0     }));
  PCAS_CHECK(block_index_info(bs        , bs * 4     , 4) == (block_info{1, bs    , bs * 2     , 0     }));
  PCAS_CHECK(block_index_info(bs * 2    , bs * 4     , 4) == (block_info{2, bs * 2, bs * 3     , 0     }));
  PCAS_CHECK(block_index_info(bs * 3    , bs * 4     , 4) == (block_info{3, bs * 3, bs * 4     , 0     }));
  PCAS_CHECK(block_index_info(bs * 4 - 1, bs * 4     , 4) == (block_info{3, bs * 3, bs * 4     , bs - 1}));
  PCAS_CHECK(block_index_info(0         , bs * 12    , 4) == (block_info{0, 0     , bs * 3     , 0     }));
  PCAS_CHECK(block_index_info(bs        , bs * 12    , 4) == (block_info{0, 0     , bs * 3     , bs    }));
  PCAS_CHECK(block_index_info(bs * 3    , bs * 12    , 4) == (block_info{1, bs * 3, bs * 6     , 0     }));
  PCAS_CHECK(block_index_info(bs * 11   , bs * 12 - 1, 4) == (block_info{3, bs * 9, bs * 12 - 1, bs * 2}));
}

template <uint64_t BlockSize>
class cyclic : public base {
  size_t block_size_;

public:
  cyclic(uint64_t size, int nproc, size_t block_size = BlockSize) : base(size, nproc), block_size_(block_size) {
    PCAS_CHECK(block_size >= BlockSize);
    PCAS_CHECK(block_size % BlockSize == 0);
  }

  uint64_t get_local_size(int rank [[maybe_unused]]) {
    uint64_t nblock_g = (size_ + block_size_ - 1) / block_size_;
    uint64_t nblock_l = (nblock_g + nproc_ - 1) / nproc_;
    return nblock_l * block_size_;
  }

  uint64_t get_effective_size() {
    return get_local_size(0) * nproc_;
  }

  block_info get_block_info(uint64_t offset) {
    PCAS_CHECK(offset < get_effective_size());
    uint64_t block_num_g = offset / block_size_;
    uint64_t block_num_l = block_num_g / nproc_;
    int      owner       = block_num_g % nproc_;
    uint64_t offset_b    = block_num_g * block_size_;
    uint64_t offset_e    = std::min((block_num_g + 1) * block_size_, size_);
    uint64_t pm_offset   = block_num_l * block_size_ + offset % block_size_;
    return block_info{owner, offset_b, offset_e, pm_offset};
  }
};

PCAS_TEST_CASE("[pcas::mem_mapper::cyclic] calculate local block size") {
  constexpr uint64_t mb = 65536;
  uint64_t bs = mb * 2;
  auto local_block_size = [=](uint64_t size, int nproc) -> uint64_t {
    return cyclic<mb>(size, nproc, bs).get_local_size(0);
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

PCAS_TEST_CASE("[pcas::mem_mapper::cyclic] get block information at specified offset") {
  constexpr uint64_t mb = 65536;
  uint64_t bs = mb * 2;
  auto block_index_info = [=](uint64_t offset, uint64_t size, int nproc) -> block_info {
    return cyclic<mb>(size, nproc, bs).get_block_info(offset);
  };
  PCAS_CHECK(block_index_info(0         , bs * 4     , 4) == (block_info{0, 0      , bs         , 0     }));
  PCAS_CHECK(block_index_info(bs        , bs * 4     , 4) == (block_info{1, bs     , bs * 2     , 0     }));
  PCAS_CHECK(block_index_info(bs * 2    , bs * 4     , 4) == (block_info{2, bs * 2 , bs * 3     , 0     }));
  PCAS_CHECK(block_index_info(bs * 3    , bs * 4     , 4) == (block_info{3, bs * 3 , bs * 4     , 0     }));
  PCAS_CHECK(block_index_info(bs * 4 - 1, bs * 4     , 4) == (block_info{3, bs * 3 , bs * 4     , bs - 1}));
  PCAS_CHECK(block_index_info(0         , bs * 12    , 4) == (block_info{0, 0      , bs         , 0     }));
  PCAS_CHECK(block_index_info(bs        , bs * 12    , 4) == (block_info{1, bs     , bs * 2     , 0     }));
  PCAS_CHECK(block_index_info(bs * 3    , bs * 12    , 4) == (block_info{3, bs * 3 , bs * 4     , 0     }));
  PCAS_CHECK(block_index_info(bs * 5 + 2, bs * 12    , 4) == (block_info{1, bs * 5 , bs * 6     , bs + 2}));
  PCAS_CHECK(block_index_info(bs * 11   , bs * 12 - 1, 4) == (block_info{3, bs * 11, bs * 12 - 1, bs * 2}));
}

}
}
