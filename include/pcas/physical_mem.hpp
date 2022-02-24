#pragma once

#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "pcas/util.hpp"

namespace pcas {

class physical_mem {
  int      fd_           = -1;
  uint64_t size_         = 0;
  void*    anon_vm_addr_ = nullptr;

public:
  physical_mem() {}
  physical_mem(uint64_t size) : size_(size) {
    fd_ = memfd_create("PCAS", 0);
    if (fd_ == -1) {
      perror("memfd_create");
      die("[pcas::physical_mem] memfd_create() failed");
    }

    if (ftruncate(fd_, size) == -1) {
      perror("ftruncate");
      die("[pcas::physical_mem] ftruncate(%d, %ld) failed", fd_, size);
    }

    anon_vm_addr_ = map(nullptr, 0, size);
  }

  physical_mem(const physical_mem&) = delete;

  physical_mem(physical_mem&& pm) :
    fd_(pm.fd_), size_(pm.size_), anon_vm_addr_(pm.anon_vm_addr_) { pm.fd_ = -1; };

  ~physical_mem() {
    if (fd_ != -1) {
      unmap(anon_vm_addr_, size_);
      close(fd_);
    }
  }

  physical_mem& operator=(const physical_mem&) = delete;

  physical_mem& operator=(physical_mem&& pm) {
    this->~physical_mem();
    fd_ = pm.fd_;
    size_ = pm.size_;
    anon_vm_addr_ = pm.anon_vm_addr_;
    pm.fd_ = -1;
    return *this;
  }

  void* map(void* addr, uint64_t offset, uint64_t size) const {
    PCAS_CHECK(offset + size <= size_);
    int flags = MAP_SHARED;
    if (addr != nullptr) flags |= MAP_FIXED;
    void* ret = mmap(addr, size, PROT_WRITE, flags, fd_, offset);
    if (ret == MAP_FAILED) {
      perror("mmap");
      die("[pcas::physical_mem] mmap(%p, %ld, ...) failed", addr, size);
    }
    return ret;
  }

  void unmap(void* addr, uint64_t size) const {
    if (munmap(addr, size) == -1) {
      perror("munmap");
      die("[pcas::physical_mem] munmap(%p, %ld) failed", addr, size);
    }
  }

  void* anon_vm_addr() const { return anon_vm_addr_; };

};

PCAS_TEST_CASE("[pcas::physical_mem] map two virtual addresses to the same physical address") {
  physical_mem pm(16 * 4096);
  int* b1 = nullptr;
  int* b2 = nullptr;

  PCAS_SUBCASE("map to random address") {
    b1 = (int*)pm.map(nullptr, 3 * 4096, 4096);
    b2 = (int*)pm.map(nullptr, 3 * 4096, 4096);
  }

  PCAS_SUBCASE("map to specified address") {
    int* tmp1 = (int*)pm.map(nullptr, 0, 4096); // get an available address
    int* tmp2 = (int*)pm.map(nullptr, 0, 4096); // get an available address
    pm.unmap(tmp1, 4096);
    pm.unmap(tmp2, 4096);
    b1 = (int*)pm.map(tmp1, 3 * 4096, 4096);
    b2 = (int*)pm.map(tmp2, 3 * 4096, 4096);
  }

  PCAS_SUBCASE("use anonymous virtual address") {
    b1 = (int*)pm.map(nullptr, 0, 16 * 4096);
    b2 = (int*)pm.anon_vm_addr();
  }

  PCAS_CHECK(b1 != b2);
  PCAS_CHECK(b1[0] == 0);
  PCAS_CHECK(b2[0] == 0);
  b1[0] = 417;
  PCAS_CHECK(b1[0] == 417);
  PCAS_CHECK(b2[0] == 417);

  pm.unmap(b1, 4096);
  pm.unmap(b2, 4096);
}

}
