#pragma once

#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "doctest/doctest.h"

namespace pcas {

class physical_mem {
  int fd_;

public:
  physical_mem(uint64_t size) {
    fd_ = memfd_create("PCAS", 0);
    if (fd_ == -1) {
      perror("memfd_create");
      exit(1);
    }

    if (ftruncate(fd_, size) == -1) {
      perror("ftruncate");
      exit(1);
    }
  }

  ~physical_mem() {
    close(fd_);
  }

  void* map(void* addr, uint64_t offset, uint64_t size) {
    int flags = MAP_SHARED;
    if (addr != nullptr) flags |= MAP_FIXED;
    void* ret = mmap(addr, size, PROT_WRITE, flags, fd_, offset);
    if (ret == MAP_FAILED) {
      perror("mmap");
      exit(1);
    }
    return ret;
  }

  void unmap(void* addr, uint64_t size) {
    if (munmap(addr, size) == -1) {
      perror("munmap");
      exit(1);
    }
  }

};

TEST_CASE("Map two virtual addresses to the same physical address") {
  physical_mem pm(16 * 4096);
  int* b1 = nullptr;
  int* b2 = nullptr;

  SUBCASE("Map to random address") {
    b1 = (int*)pm.map(nullptr, 3 * 4096, 4096);
    b2 = (int*)pm.map(nullptr, 3 * 4096, 4096);
  }

  SUBCASE("Map to specified address") {
    int* tmp1 = (int*)pm.map(nullptr, 0, 4096); // get an available address
    int* tmp2 = (int*)pm.map(nullptr, 0, 4096); // get an available address
    pm.unmap(tmp1, 4096);
    pm.unmap(tmp2, 4096);
    b1 = (int*)pm.map(tmp1, 3 * 4096, 4096);
    b2 = (int*)pm.map(tmp2, 3 * 4096, 4096);
  }

  CHECK(b1 != b2);
  CHECK(b1[0] == 0);
  CHECK(b2[0] == 0);
  b1[0] = 417;
  CHECK(b1[0] == 417);
  CHECK(b2[0] == 417);

  pm.unmap(b1, 4096);
  pm.unmap(b2, 4096);
}

}
