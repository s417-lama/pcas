#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include "pcas/pcas.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  doctest::Context ctx;
  ctx.setOption("abort-after", 5);
  ctx.setOption("force-colors", true);
  ctx.applyCommandLine(argc, argv);

  int test_result = ctx.run();

  MPI_Finalize();

  return test_result;
}
