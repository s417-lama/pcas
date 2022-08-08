#pragma once

#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "pcas/util.hpp"
#include "pcas/physical_mem.hpp"

namespace pcas {

class virtual_mem {
  void* addr_ = nullptr;
  uint64_t size_;

public:
  virtual_mem() {}
  virtual_mem(void* addr, uint64_t size) : size_(size) {
    addr_ = mmap_no_physical_mem(addr, size);
  }

  virtual_mem(const virtual_mem&) = delete;

  virtual_mem(virtual_mem&& vm) : addr_(vm.addr_), size_(vm.size_) { vm.addr_ = nullptr; }

  ~virtual_mem() {
    if (addr_) {
      unmap(addr_, size_);
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

  void* map_physical_mem(uint64_t vm_offset, uint64_t pm_offset, uint64_t size, physical_mem& pm) const {
    PCAS_CHECK(vm_offset + size <= size_);
    void* ret = pm.map((uint8_t*)addr_ + vm_offset, pm_offset, size);
    PCAS_CHECK(ret == (uint8_t*)addr_ + vm_offset);
    return ret;
  }

  void unmap_physical_mem(uint64_t vm_offset, uint64_t size) const {
    void* ret = mmap_no_physical_mem((uint8_t*)addr_ + vm_offset, size);
    PCAS_CHECK(ret == (uint8_t*)addr_ + vm_offset);
  }

  // TODO: reconsider this abstraction...
  static void* mmap_no_physical_mem(void* addr, uint64_t size) {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (addr != nullptr) flags |= MAP_FIXED;
    void* ret = mmap(addr, size, PROT_NONE, flags, -1, 0);
    if (ret == MAP_FAILED) {
      perror("mmap");
      die("[pcas::virtual_mem] mmap(%p, %lu, ...) failed", addr, size);
    }
    return ret;
  }

  static void unmap(void* addr, uint64_t size) {
    if (munmap(addr, size) == -1) {
      perror("munmap");
      die("[pcas::virtual_mem] munmap(%p, %lu) failed", addr, size);
    }
  }

};

PCAS_TEST_CASE("[pcas::virtual_mem] allocate virtual memory") {
  uint64_t pagesize = sysconf(_SC_PAGE_SIZE);

  void* addr = nullptr;
  {
    virtual_mem vm(nullptr, 32 * pagesize);
    PCAS_CHECK(vm.addr() != nullptr);
    addr = vm.addr();
  }
  {
    virtual_mem vm(addr, 16 * pagesize);
    PCAS_CHECK(vm.addr() == addr);
  }
}

PCAS_TEST_CASE("[pcas::virtual_mem] map physical memory to virtual memory") {
  uint64_t pagesize = sysconf(_SC_PAGE_SIZE);

  virtual_mem vm(nullptr, 20 * pagesize);
  physical_mem pm(10 * pagesize, 0, -1, true, true);
  uint8_t* mapped_addr = (uint8_t*)vm.map_physical_mem(5 * pagesize, 5 * pagesize, 5 * pagesize, pm);
  PCAS_CHECK(mapped_addr == (uint8_t*)vm.addr() + 5 * pagesize);

  uint8_t* pm_vals = (uint8_t*)pm.map(nullptr, 0, 10 * pagesize);
  pm_vals[5 * pagesize] = 17;
  PCAS_CHECK(mapped_addr[0] == 17);
  mapped_addr[1] = 32;
  PCAS_CHECK(pm_vals[5 * pagesize + 1] == 32);
}

}
