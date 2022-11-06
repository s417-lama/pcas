#pragma once

#include <vector>
#include <memory>
#include <mpi.h>

#include "pcas/util.hpp"
#include "pcas/global_ptr.hpp"
#include "pcas/virtual_mem.hpp"
#include "pcas/physical_mem.hpp"
#include "pcas/mem_mapper.hpp"
#include "pcas/topology.hpp"

namespace pcas {

using mem_obj_id_t = uint64_t;
using mem_block_num_t = uint64_t;

template <typename P>
class mem_obj_if {
  std::unique_ptr<mem_mapper::base> mmapper_;
  mem_obj_id_t                      id_;
  uint64_t                          size_;
  const topology&                   topo_;
  uint64_t                          local_size_;
  uint64_t                          effective_size_;
  virtual_mem                       vm_;
  std::vector<physical_mem>         home_pms_; // intra-rank -> pm
  win_manager                       win_;
  mem_block_num_t                   last_checkout_block_num_ = std::numeric_limits<mem_block_num_t>::max();

  static bool num_prefetch_blocks(topology::rank_t global_rank) {
    static bool n_prefetch_ = get_env("PCAS_PREFETCH_BLOCKS", 0, global_rank);
    return n_prefetch_;
  }

  static std::string home_shmem_name(mem_obj_id_t id, int global_rank) {
    std::stringstream ss;
    ss << "/pcas_" << id << "_" << global_rank;
    return ss.str();
  }

  std::vector<physical_mem> init_home_pms() const {
    physical_mem pm_local(home_shmem_name(id_, topo_.global_rank()), local_size_, true, true);

    MPI_Barrier(topo_.intra_comm());

    // Open home physical memory of other intra-node processes
    std::vector<physical_mem> home_pms(topo_.intra_nproc());
    for (int i = 0; i < topo_.intra_nproc(); i++) {
      if (i == topo_.intra_rank()) {
        home_pms[i] = std::move(pm_local);
      } else {
        int target_rank = topo_.intra2global_rank(i);
        int target_local_size = mmapper_->get_local_size(target_rank);
        physical_mem pm(home_shmem_name(id_, target_rank), target_local_size, false, true);
        home_pms[i] = std::move(pm);
      }
    }

    return home_pms;
  }

public:
  mem_obj_if(std::unique_ptr<mem_mapper::base> mmapper,
             mem_obj_id_t id,
             uint64_t size,
             const topology& topo) :
    mmapper_(std::move(mmapper)),
    id_(id),
    size_(size),
    topo_(topo),
    local_size_(mmapper_->get_local_size(topo_.global_rank())),
    effective_size_(mmapper_->get_effective_size()),
    vm_(reserve_same_vm_coll(topo_.global_comm(), effective_size_, P::block_size)),
    home_pms_(init_home_pms()),
    win_(topo_.global_comm(), home_pm().anon_vm_addr(), local_size_) {}

  const mem_mapper::base& mem_mapper() const { return *mmapper_; }

  mem_obj_id_t id() const { return id_; }
  uint64_t size() const { return size_; }
  uint64_t local_size() const { return local_size_; }
  uint64_t effective_size() const { return effective_size_; }

  const virtual_mem& vm() const { return vm_; }

  const physical_mem& home_pm() const {
    return home_pms_[topo_.intra_rank()];
  }

  const physical_mem& home_pm(topology::rank_t intra_rank) const {
    PCAS_CHECK(intra_rank < topo_.intra_nproc());
    return home_pms_[intra_rank];
  }

  MPI_Win win() const { return win_.win(); }

  uint64_t size_with_prefetch(uint64_t offset, uint64_t size) {
    uint64_t size_pf = size;
    uint64_t n_prefetch = num_prefetch_blocks(topo_.global_rank());
    if (n_prefetch > 0) {
      mem_block_num_t block_num_b = offset / P::block_size;
      mem_block_num_t block_num_e = (offset + size + P::block_size - 1) / P::block_size;
      if (block_num_b <= last_checkout_block_num_ + 1 &&
          last_checkout_block_num_ + 1 < block_num_e) {
        // If it seems sequential access, do prefetch
        size_pf = std::min(size + n_prefetch * P::block_size, size_ - offset);
      }
      last_checkout_block_num_ = block_num_e - 1;
    }
    return size_pf;
  }

};

struct mem_obj_policy_default {
  constexpr static uint64_t block_size = 65536;
};

}
