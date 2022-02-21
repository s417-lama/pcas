#pragma once

#include <vector>

#include <mpi.h>

#include "doctest/doctest.h"

#include "pcas/util.hpp"

namespace pcas {

class pcas {
private:
  int rank_;
  int nprocs_;

public:
  pcas();
  ~pcas();
};

pcas::pcas() {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    die("MPI_Init() must be called before initializing PCAS.\n");
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs_);

  MPI_Barrier(MPI_COMM_WORLD);
}

pcas::~pcas() {
}

TEST_CASE("Initialize and Finalize PCAS") {
  for (int i = 0; i < 3; i++) {
    pcas pc;
  }
}

}
