#pragma once

#include <cstdint>
#include <mpi.h>

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

}
