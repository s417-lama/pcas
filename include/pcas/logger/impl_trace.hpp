#pragma once

#include <cstdint>
#include <sstream>

#include <mpi.h>

/* #define MLOG_DISABLE_CHECK_BUFFER_SIZE 1 */
/* #define MLOG_DISABLE_REALLOC_BUFFER    1 */
#include "mlog/mlog.h"

#include "pcas/util.hpp"

namespace pcas {
namespace logger {

template <typename P>
class impl_trace {
public:
  using begin_data_t = void*;

private:
  using this_t = impl_trace;
  using kind = typename P::logger_kind_t;
  using wallclock = typename P::wallclock_t;

  int rank_;
  int n_ranks_;
  FILE* stream_;

  uint64_t t_begin_;
  uint64_t t_end_;

  bool     stat_print_per_rank_;
  uint64_t stat_acc_[kind::size()];
  uint64_t stat_acc_total_[kind::size()];
  uint64_t stat_count_[kind::size()];
  uint64_t stat_count_total_[kind::size()];

  mlog_data_t md_;

  static this_t& get_instance_() {
    static this_t my_instance;
    return my_instance;
  }

  template <typename kind::value K>
  static void* logger_decoder_tl_(FILE* stream, int _rank0, int _rank1, void* buf0, void* buf1) {
    this_t& lgr = get_instance_();

    uint64_t t0 = MLOG_READ_ARG(&buf0, uint64_t);
    uint64_t t1 = MLOG_READ_ARG(&buf1, uint64_t);

    if (t1 < lgr.t_begin_ || lgr.t_end_ < t0) {
      return buf1;
    }

    acc_stat_<K>(t0, t1);

    fprintf(stream, "%d,%lu,%d,%lu,%s\n", lgr.rank_, t0, lgr.rank_, t1, kind(K).str());
    return buf1;
  }

  template <typename kind::value K, typename Misc>
  static void* logger_decoder_tl_w_misc_(FILE* stream, int _rank0, int _rank1, void* buf0, void* buf1) {
    this_t& lgr = get_instance_();

    uint64_t t0 = MLOG_READ_ARG(&buf0, uint64_t);
    uint64_t t1 = MLOG_READ_ARG(&buf1, uint64_t);
    Misc     m  = MLOG_READ_ARG(&buf1, Misc);

    if (t1 < lgr.t_begin_ || lgr.t_end_ < t0) {
      return buf1;
    }

    acc_stat_<K>(t0, t1);

    std::stringstream ss;
    ss << m;
    fprintf(stream, "%d,%lu,%d,%lu,%s,%s\n", lgr.rank_, t0, lgr.rank_, t1, kind(K).str(), ss.str().c_str());
    return buf1;
  }

  static void print_kind_stat_(kind k, int rank) {
    if (k.is_valid()) {
      this_t& lgr = get_instance_();
      if (lgr.stat_print_per_rank_) {
        uint64_t acc = lgr.stat_acc_[k.index()];
        uint64_t acc_total = lgr.t_end_ - lgr.t_begin_;
        uint64_t count = lgr.stat_count_[k.index()];
        printf("(Rank %3d) %-23s : %10.6f %% ( %15ld ns / %15ld ns ) count: %8ld\n",
               rank, k.str(), (double)acc / acc_total * 100, acc, acc_total, count);
      } else {
        uint64_t acc = lgr.stat_acc_total_[k.index()];
        uint64_t acc_total = (lgr.t_end_ - lgr.t_begin_) * lgr.n_ranks_;
        uint64_t count = lgr.stat_count_total_[k.index()];
        printf("  %-23s : %10.6f %% ( %15ld ns / %15ld ns ) count: %8ld\n",
               k.str(), (double)acc / acc_total * 100, acc, acc_total, count);
      }
    }
  }

  template <typename kind::value K>
  static void acc_stat_(uint64_t t0, uint64_t t1) {
    this_t& lgr = get_instance_();
    uint64_t t0_ = std::max(t0, lgr.t_begin_);
    uint64_t t1_ = std::min(t1, lgr.t_end_);
    if (t1_ > t0_) {
      lgr.stat_acc_[kind(K).index()] += t1_ - t0_;
      lgr.stat_count_[kind(K).index()]++;
    }
  }

  static void print_stat_(int rank) {
    for (size_t k = 1; k < kind::size(); k++) {
      print_kind_stat_(kind((typename kind::value)k), rank);
    }
    printf("\n");
  }

public:
  static void init(int rank, int n_ranks) {
    size_t size = get_env("PCAS_LOGGER_INITIAL_SIZE", 1 << 20, rank);

    this_t& lgr = get_instance_();
    lgr.rank_ = rank;
    lgr.n_ranks_ = n_ranks;

    lgr.stat_print_per_rank_ = get_env("PCAS_LOGGER_PRINT_STAT_PER_RANK", false, rank);

    mlog_init(&lgr.md_, 1, size);

    wallclock::init();
    wallclock::sync();

    char filename[128];
    sprintf(filename, "pcas_log_%d.ignore", rank);
    lgr.stream_ = fopen(filename, "w+");
  }

  static void flush(uint64_t t_begin, uint64_t t_end) {
    this_t& lgr = get_instance_();

    lgr.t_begin_ = t_begin;
    lgr.t_end_ = t_end;

    mlog_flush_all(&lgr.md_, lgr.stream_);
  }

  static void flush_and_print_stat(uint64_t t_begin, uint64_t t_end) {
    this_t& lgr = get_instance_();

    for (size_t k = 0; k < kind::size(); k++) {
      lgr.stat_acc_[k] = 0;
      lgr.stat_acc_total_[k] = 0;
      lgr.stat_count_[k] = 0;
      lgr.stat_count_total_[k] = 0;
    }

    flush(t_begin, t_end);

    if (lgr.stat_print_per_rank_) {
      if (lgr.rank_ == 0) {
        print_stat_(0);
        for (int i = 1; i < lgr.n_ranks_; i++) {
          MPI_Recv(lgr.stat_acc_, kind::size(), MPI_UINT64_T,
                   i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(lgr.stat_count_, kind::size(), MPI_UINT64_T,
                   i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          print_stat_(i);
        }
      } else {
        MPI_Send(lgr.stat_acc_, kind::size(), MPI_UINT64_T,
                 0, 0, MPI_COMM_WORLD);
        MPI_Send(lgr.stat_count_, kind::size(), MPI_UINT64_T,
                 0, 0, MPI_COMM_WORLD);
      }
    } else {
      MPI_Reduce(lgr.stat_acc_, lgr.stat_acc_total_, kind::size(),
                 MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(lgr.stat_count_, lgr.stat_count_total_, kind::size(),
                 MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
      if (lgr.rank_ == 0) {
        print_stat_(0);
      }
    }
  }

  static void warmup() {
    this_t& lgr = get_instance_();
    mlog_warmup(&lgr.md_, 0);
  }

  static void clear() {
    this_t& lgr = get_instance_();
    mlog_clear_all(&lgr.md_);
  }

  template <typename kind::value K>
  static begin_data_t begin_event() {
    if (kind(K).is_valid()) {
      this_t& lgr = get_instance_();
      uint64_t t = wallclock::get_time();
      begin_data_t bd = MLOG_BEGIN(&lgr.md_, 0, t);
      return bd;
    } else {
      return nullptr;
    }
  }

  template <typename kind::value K>
  static void end_event(begin_data_t bd) {
    if (kind(K).is_valid()) {
      this_t& lgr = get_instance_();
      uint64_t t = wallclock::get_time();
      auto fn = &logger_decoder_tl_<K>;
      MLOG_END(&lgr.md_, 0, bd, fn, t);
    }
  }

  template <typename kind::value K, typename Misc>
  static void end_event(begin_data_t bd, Misc m) {
    if (kind(K).is_valid()) {
      this_t& lgr = get_instance_();
      uint64_t t = wallclock::get_time();
      auto fn = &logger_decoder_tl_w_misc_<K, Misc>;
      MLOG_END(&lgr.md_, 0, bd, fn, t, m);
    }
  }
};

}
}
