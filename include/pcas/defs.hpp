#pragma once

#include <cstdint>
#include <mpi.h>

#include "pcas/global_ptr.hpp"

namespace pcas {

enum class dist_policy {
  local,
  block,
  block_cyclic,
};

using obj_id_t = uint64_t;

struct obj_entry {
  int         owner;
  obj_id_t    id;
  uint64_t    size;
  dist_policy dpolicy;
  uint64_t    block_size;
  uint8_t*    baseptr;
  MPI_Win     win;
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

}
