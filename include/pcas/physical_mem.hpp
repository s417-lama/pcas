#pragma once

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <sstream>

#include "pcas/util.hpp"

namespace pcas {

class physical_mem {
  int         fd_           = -1;
  uint64_t    size_         = 0;
  void*       anon_vm_addr_ = nullptr;
  bool        own_;
  bool        map_anon_;
  std::string shm_name_;

public:
  physical_mem() {}
  physical_mem(uint64_t size, obj_id_t id, int intra_rank, bool own, bool map_anon)
    : size_(size), own_(own), map_anon_(map_anon) {
    if (intra_rank == -1) {
      intra_rank = getpid(); // for testing
    }

    std::stringstream ss;
    ss << "/pcas_" << id << "_" << intra_rank;
    shm_name_ = ss.str();

    int oflag = O_RDWR;
    if (own) oflag |= O_CREAT | O_TRUNC;

    fd_ = shm_open(shm_name_.c_str(), oflag, S_IRUSR | S_IWUSR);
    if (fd_ == -1) {
      perror("shm_open");
      die("[pcas::physical_mem] shm_open() failed");
    }

    if (own && ftruncate(fd_, size) == -1) {
      perror("ftruncate");
      die("[pcas::physical_mem] ftruncate(%d, %lu) failed", fd_, size);
    }

    if (map_anon) {
      anon_vm_addr_ = map(nullptr, 0, size);
    }
  }

  physical_mem(const physical_mem&) = delete;

  physical_mem(physical_mem&& pm)
    : fd_(pm.fd_), size_(pm.size_), anon_vm_addr_(pm.anon_vm_addr_),
      own_(pm.own_), map_anon_(pm.map_anon_), shm_name_(std::move(pm.shm_name_)) {
    pm.fd_ = -1;
  }

  ~physical_mem() {
    if (fd_ != -1) {
      if (map_anon_) {
        unmap(anon_vm_addr_, size_);
      }
      close(fd_);
      if (own_ && shm_unlink(shm_name_.c_str()) == -1) {
        perror("shm_unlink");
        die("[pcas::physical_mem] shm_unlink() failed");
      }
    }
  }

  physical_mem& operator=(const physical_mem&) = delete;

  physical_mem& operator=(physical_mem&& pm) {
    this->~physical_mem();
    fd_ = pm.fd_;
    size_ = pm.size_;
    anon_vm_addr_ = pm.anon_vm_addr_;
    own_ = pm.own_;
    map_anon_ = pm.map_anon_;
    shm_name_ = std::move(pm.shm_name_);
    pm.fd_ = -1;
    return *this;
  }

  void* map(void* addr, uint64_t offset, uint64_t size) const {
    PCAS_CHECK(offset + size <= size_);
    int flags = MAP_SHARED;
    if (addr != nullptr) flags |= MAP_FIXED;
    void* ret = mmap(addr, size, PROT_READ | PROT_WRITE, flags, fd_, offset);
    if (ret == MAP_FAILED) {
      perror("mmap");
      die("[pcas::physical_mem] mmap(%p, %lu, ...) failed", addr, size);
    }
    return ret;
  }

  void unmap(void* addr, uint64_t size) const {
    if (munmap(addr, size) == -1) {
      perror("munmap");
      die("[pcas::physical_mem] munmap(%p, %lu) failed", addr, size);
    }
  }

  void* anon_vm_addr() const { return anon_vm_addr_; };

};

PCAS_TEST_CASE("[pcas::physical_mem] map two virtual addresses to the same physical address") {
  uint64_t pagesize = sysconf(_SC_PAGE_SIZE);

  physical_mem pm(16 * pagesize, 0, -1, true, true);
  int* b1 = nullptr;
  int* b2 = nullptr;

  PCAS_SUBCASE("map to random address") {
    b1 = (int*)pm.map(nullptr, 3 * pagesize, pagesize);
    b2 = (int*)pm.map(nullptr, 3 * pagesize, pagesize);
  }

  PCAS_SUBCASE("map to specified address") {
    int* tmp1 = (int*)pm.map(nullptr, 0, pagesize); // get an available address
    int* tmp2 = (int*)pm.map(nullptr, 0, pagesize); // get an available address
    pm.unmap(tmp1, pagesize);
    pm.unmap(tmp2, pagesize);
    b1 = (int*)pm.map(tmp1, 3 * pagesize, pagesize);
    b2 = (int*)pm.map(tmp2, 3 * pagesize, pagesize);
  }

  PCAS_SUBCASE("use anonymous virtual address") {
    b1 = (int*)pm.map(nullptr, 0, 16 * pagesize);
    b2 = (int*)pm.anon_vm_addr();
  }

  PCAS_CHECK(b1 != b2);
  PCAS_CHECK(b1[0] == 0);
  PCAS_CHECK(b2[0] == 0);
  b1[0] = 417;
  PCAS_CHECK(b1[0] == 417);
  PCAS_CHECK(b2[0] == 417);

  pm.unmap(b1, pagesize);
  pm.unmap(b2, pagesize);
}

}
