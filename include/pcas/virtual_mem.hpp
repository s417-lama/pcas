#pragma once

#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <mpi.h>

#include "pcas/util.hpp"
#include "pcas/physical_mem.hpp"

namespace pcas {

class mmap_noreplace_exception : public std::exception {};

class virtual_mem {
  void* addr_ = nullptr;
  std::size_t size_;

public:
  virtual_mem() {}
  virtual_mem(void* addr, std::size_t size, std::size_t alignment = alignof(max_align_t)) :
    addr_(mmap_no_physical_mem(addr, size, false, alignment)), size_(size) {}

  ~virtual_mem() {
    if (addr_) {
      unmap(addr_, size_);
    }
  }

  virtual_mem(const virtual_mem&) = delete;
  virtual_mem& operator=(const virtual_mem&) = delete;

  virtual_mem(virtual_mem&& vm) : addr_(vm.addr_), size_(vm.size_) { vm.addr_ = nullptr; }
  virtual_mem& operator=(virtual_mem&& vm) {
    this->~virtual_mem();
    addr_ = vm.addr();
    size_ = vm.size();
    vm.addr_ = nullptr;
    return *this;
  }

  void* addr() const { return addr_; }
  std::size_t size() const { return size_; }

  void* map_physical_mem(std::size_t vm_offset, std::size_t pm_offset, std::size_t size, physical_mem& pm) const {
    PCAS_CHECK(vm_offset + size <= size_);
    void* ret = pm.map(reinterpret_cast<std::byte*>(addr_) + vm_offset, pm_offset, size);
    PCAS_CHECK(ret == reinterpret_cast<std::byte*>(addr_) + vm_offset);
    return ret;
  }

  void unmap_physical_mem(std::size_t vm_offset, std::size_t size) const {
    void* ret = mmap_no_physical_mem(reinterpret_cast<std::byte*>(addr_) + vm_offset, size);
    PCAS_CHECK(ret == reinterpret_cast<std::byte*>(addr_) + vm_offset);
  }

  // TODO: reconsider this abstraction...
  static void* mmap_no_physical_mem(void*       addr,
                                    std::size_t size,
                                    bool        replace   = true,
                                    std::size_t alignment = alignof(max_align_t)) {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;

    std::size_t reqsize;
    if (addr == nullptr) {
      reqsize = size + alignment;
    } else {
      PCAS_CHECK(reinterpret_cast<uintptr_t>(addr) % alignment == 0);
      reqsize = size;
      if (!replace) {
        flags |= MAP_FIXED_NOREPLACE;
      } else {
        flags |= MAP_FIXED;
      }
    }

    void* allocated_p = mmap(addr, reqsize, PROT_NONE, flags, -1, 0);
    if (allocated_p == MAP_FAILED) {
      if (errno == EEXIST) {
        // MAP_FIXED_NOREPLACE error
        throw mmap_noreplace_exception{};
      } else {
        perror("mmap");
        die("[pcas::virtual_mem] mmap(%p, %lu, ...) failed", addr, reqsize);
      }
    }

    if (addr == nullptr) {
      uintptr_t allocated_addr = reinterpret_cast<uintptr_t>(allocated_p);
      uintptr_t ret_addr = round_up_pow2(allocated_addr, alignment);
      std::byte* ret_p = reinterpret_cast<std::byte*>(ret_addr);

      PCAS_CHECK(ret_addr >= allocated_addr);
      unmap(allocated_p, ret_addr - allocated_addr);

      PCAS_CHECK(reqsize >= ret_addr - allocated_addr + size);
      unmap(ret_p + size, reqsize - (ret_addr - allocated_addr + size));

      return ret_p;
    } else {
      PCAS_CHECK(addr == allocated_p);
      return allocated_p;
    }
  }

  static void unmap(void* addr, std::size_t size) {
    if (size > 0 && munmap(addr, size) == -1) {
      perror("munmap");
      die("[pcas::virtual_mem] munmap(%p, %lu) failed", addr, size);
    }
  }

};

PCAS_TEST_CASE("[pcas::virtual_mem] allocate virtual memory") {
  std::size_t pagesize = sysconf(_SC_PAGE_SIZE);

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
  std::size_t pagesize = sysconf(_SC_PAGE_SIZE);

  std::stringstream ss;
  ss << "/pcas_test_" << getpid();

  virtual_mem vm(nullptr, 20 * pagesize);
  physical_mem pm(ss.str(), 10 * pagesize, true, true);
  uint8_t* mapped_addr = (uint8_t*)vm.map_physical_mem(5 * pagesize, 5 * pagesize, 5 * pagesize, pm);
  PCAS_CHECK(mapped_addr == (uint8_t*)vm.addr() + 5 * pagesize);

  uint8_t* pm_vals = (uint8_t*)pm.map(nullptr, 0, 10 * pagesize);
  pm_vals[5 * pagesize] = 17;
  PCAS_CHECK(mapped_addr[0] == 17);
  mapped_addr[1] = 32;
  PCAS_CHECK(pm_vals[5 * pagesize + 1] == 32);
}

virtual_mem reserve_same_vm_coll(MPI_Comm comm, std::size_t size, std::size_t alignment = alignof(max_align_t)) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  uintptr_t vm_addr;
  virtual_mem vm;

  std::vector<virtual_mem> prev_vms;
  int max_trial = 100;

  // Repeat until the same virtual memory address is allocated
  // TODO: smarter allocation using `pmap` result?
  for (int n_trial = 0; n_trial <= max_trial; n_trial++) {
    if (rank == 0) {
      vm = virtual_mem(nullptr, size, alignment);
      vm_addr = reinterpret_cast<uintptr_t>(vm.addr());
    }

    MPI_Bcast(&vm_addr, 1, MPI_UINT64_T, 0, comm);

    int status = 0; // 0: OK, 1: failed
    if (rank != 0) {
      try {
        vm = virtual_mem(reinterpret_cast<void*>(vm_addr), size, alignment);
      } catch (mmap_noreplace_exception& e) {
        status = 1;
      }
    }

    int status_all = 0;
    MPI_Allreduce(&status, &status_all, 1, MPI_INT, MPI_SUM, comm);

    if (status_all == 0) {
      // prev_vms are automatically freed
      return vm;
    }

    if (status == 0) {
      // Defer the deallocation of previous virtual addresses to prevent
      // the same address from being allocated next
      prev_vms.push_back(std::move(vm));
    }
  }

  die("Reservation of virtual memory address failed (max_trial=%d)", max_trial);
  throw;
}

}
