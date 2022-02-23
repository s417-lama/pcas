#pragma once

#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "doctest/doctest.h"

namespace pcas {

class virtual_mem {
  void* addr_ = nullptr;
  uint64_t size_;

public:
  virtual_mem() {}
  virtual_mem(void* addr, uint64_t size) {
    int flags = MAP_SHARED | MAP_ANONYMOUS;
    if (addr != nullptr) flags |= MAP_FIXED;
    addr_ = mmap(addr, size, PROT_NONE, flags, -1, 0);
    if (addr_ == MAP_FAILED) {
      perror("mmap");
      exit(1);
    }
    size_ = size;
  }

  virtual_mem(const virtual_mem&) = delete;

  virtual_mem(virtual_mem&& vm) : addr_(vm.addr_), size_(vm.size_) { vm.addr_ = nullptr; }

  ~virtual_mem() {
    if (addr_) {
      if (munmap(addr_, size_) == -1) {
        perror("munmap");
        exit(1);
      }
    }
  }

  virtual_mem& operator=(const virtual_mem&) = delete;

  virtual_mem& operator=(virtual_mem&& vm) {
    this->~virtual_mem();
    addr_ = vm.addr();
    size_ = vm.size();
    vm.addr_ = nullptr;
    return *this;
  }

  void* addr() const { return addr_; }
  uint64_t size() const { return size_; }
};

TEST_CASE("allocate virtual memory") {
  void* addr = nullptr;
  {
    virtual_mem vm(nullptr, 32 * 4096);
    CHECK(vm.addr() != nullptr);
    addr = vm.addr();
  }
  {
    virtual_mem vm(addr, 16 * 4096);
    CHECK(vm.addr() == addr);
  }
}

}
