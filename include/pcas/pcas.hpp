#pragma once

#include <vector>

#include <mpi.h>

#include "doctest/doctest.h"

#include "pcas/util.hpp"
#include "pcas/global_ptr.hpp"

namespace pcas {

enum class dist_type {
  local,
  block,
  block_cyclic,
};

class pcas {
  int rank_;
  int nprocs_;
  MPI_Comm comm_;

public:
  pcas(MPI_Comm comm = MPI_COMM_WORLD);
  ~pcas();

  void barrier() const { MPI_Barrier(comm_); }
};

pcas::pcas(MPI_Comm comm) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    die("MPI_Init() must be called before initializing PCAS.\n");
  }

  comm_ = comm;

  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &nprocs_);

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

}
