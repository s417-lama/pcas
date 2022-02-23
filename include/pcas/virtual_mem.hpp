#pragma once

#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>

#include "doctest/doctest.h"

#include "physical_mem.hpp"

namespace pcas {

class virtual_mem {
  void* addr_ = nullptr;
  uint64_t size_;

  void mmap_no_physical_mem(void* addr, uint64_t size) {
    int flags = MAP_SHARED | MAP_ANONYMOUS;
    if (addr != nullptr) flags |= MAP_FIXED;
    addr_ = mmap(addr, size, PROT_NONE, flags, -1, 0);
    if (addr_ == MAP_FAILED) {
      perror("mmap");
      exit(1);
    }
  }

  void munmap_(void* addr, uint64_t size) {
    if (munmap(addr, size) == -1) {
      perror("munmap");
      exit(1);
    }
  }

public:
  virtual_mem() {}
  virtual_mem(void* addr, uint64_t size) {
    mmap_no_physical_mem(addr, size);
    size_ = size;
  }

  virtual_mem(const virtual_mem&) = delete;

  virtual_mem(virtual_mem&& vm) : addr_(vm.addr_), size_(vm.size_) { vm.addr_ = nullptr; }

  ~virtual_mem() {
    if (addr_) {
      munmap_(addr_, size_);
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

  void* map_physical_mem(uint64_t vm_offset, uint64_t pm_offset, uint64_t size, physical_mem& pm) {
    assert(vm_offset + size <= size_);
    munmap_((uint8_t*)addr_ + vm_offset, size); // TODO: needed?
    void* ret = pm.map((uint8_t*)addr_ + vm_offset, pm_offset, size);
    assert(ret == (uint8_t*)addr_ + vm_offset);
    return ret;
  }

  void unmap_physical_mem(uint64_t vm_offset, uint64_t size) {
    munmap_((uint8_t*)addr_ + vm_offset, size);
    mmap_no_physical_mem((uint8_t*)addr_ + vm_offset, size);
  }

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

TEST_CASE("map physical memory to virtual memory") {
  virtual_mem vm(nullptr, 20 * 4096);
  physical_mem pm(10 * 4096);
  uint8_t* mapped_addr = (uint8_t*)vm.map_physical_mem(5 * 4096, 5 * 4096, 5 * 4096, pm);
  CHECK(mapped_addr == (uint8_t*)vm.addr() + 5 * 4096);

  uint8_t* pm_vals = (uint8_t*)pm.map(nullptr, 0, 10 * 4096);
  pm_vals[5 * 4096] = 17;
  CHECK(mapped_addr[0] == 17);
  mapped_addr[1] = 32;
  CHECK(pm_vals[5 * 4096 + 1] == 32);
}

}
