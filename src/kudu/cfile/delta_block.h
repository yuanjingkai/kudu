// Copyright (c) 2013, Cloudera, inc.
// Confidential Cloudera Information: Covered by NDA.
//
// Delta encoding is adapted from the Binary packing described in
// "Decoding billions of integers per second through vectorization"
// by D. Lemire and L. Boytsov
#ifndef KUDU_CFILE_DELTA_BLOCK_H
#define KUDU_CFILE_DELTA_BLOCK_H

#include <algorithm>
#include <vector>
#include <stdint.h>

#include <boost/type_traits/make_unsigned.hpp>

#include "kudu/cfile/cfile_util.h"
#include "kudu/common/columnblock.h"
#include "kudu/util/coding.h"
#include "kudu/util/coding-inl.h"

#include "kudu/cfile/block_encodings.h"
#include "kudu/util/bit-stream-utils.h"
#include "kudu/gutil/casts.h"
#include "kudu/gutil/mathlimits.h"

namespace kudu { namespace cfile {

using std::vector;

// Delta encoding supports signed integer. (INT32, INT16, INT8)
//
// The block layout of delta encoding is defined as follows: 
// Header section
//  - ordinal_pos
//  - total value count
//  - number of miniblocks
// Metadata section
//  - list of bitwidths of miniblocks
//  - list of min deltas of miniblocks
//  - list of first values of miniblocks
// Data section
//  - list of miniblocks: each miniblock is a list of bit packed 
//    ints according to the bitwidth stored at the metadata 
//    section.
template<DataType Type>
class DeltaBlockBuilder : public BlockBuilder {
 public:
  explicit DeltaBlockBuilder(const WriterOptions* options)
    : bit_writer_(&this->buffer_),
      options_(options) {
    Reset();
  }

  void Reset() {
    num_elems_ = 0;
    deltas_.clear();
    mblocks_.clear();
    bit_writer_.Clear();
    size_estimate_ = kHeaderSizeBytes;
  }

  int Add(const uint8_t* vals_void, size_t count) {
    const CppType* vals = reinterpret_cast<const CppType*>(vals_void);
    size_t added = 0;

    while (!IsBlockFull(options_->storage_attributes.cfile_block_size) && added < count) {
      added++;
      CppType cur_elem = *vals++;

      if (num_elems_ == 0) {
        first_elem_ = cur_elem;
        cur_mblock_.num_elems = 0;
      }
      num_elems_++;

      if (cur_mblock_.num_elems == 0) {
        cur_mblock_.num_elems++;
        cur_mblock_.min_delta = std::numeric_limits<UnsignedCppType>::max();
        cur_mblock_.max_delta = std::numeric_limits<UnsignedCppType>::min();
        cur_mblock_.first_elem = cur_elem;
        last_elem_ = cur_elem;
        continue;
      } else {
        // Cast to the unsigned type first, so that integer overflow/underflow is defined.
        UnsignedCppType delta =
            static_cast<UnsignedCppType>(cur_elem) - static_cast<UnsignedCppType>(last_elem_);
        deltas_.push_back(delta);
        cur_mblock_.num_elems++;
        // Compute the min and max deltas in current miniblock.
        cur_mblock_.min_delta = std::min(delta, cur_mblock_.min_delta);
        cur_mblock_.max_delta = std::max(delta, cur_mblock_.max_delta);
      }

      if (cur_mblock_.num_elems < kEntriesPerMiniBlock) {
        size_estimate_ += size_of_type;
      } else {
        // The bit width for this miniblock is the number of bits needed to store
        // (max_delta - min_delta).
        uint64_t delta_range = cur_mblock_.max_delta - cur_mblock_.min_delta;
        cur_mblock_.bit_width = delta_range == 0 ? 0 : Bits::Log2Ceiling64(delta_range); 
        uint32_t required_bytes =
            BitUtil::Ceil(cur_mblock_.bit_width * kDeltaEntriesPerMiniBlock, kBitsPerByte);

        // Compute the number of bytes needed to store the encoded values and related metadata.
        size_estimate_ -= size_of_type * kDeltaEntriesPerMiniBlock;
        size_estimate_ += required_bytes + kOverheadPerMiniBlock;
        mblocks_.push_back(cur_mblock_);
        cur_mblock_.num_elems = 0;
      }
      last_elem_ = cur_elem;
    }
    return added;
  }

  bool IsBlockFull(size_t limit) const OVERRIDE {
    return EstimateEncodedSize() > limit;
  }

  size_t Count() const {
    return num_elems_;
  }

  Status GetFirstKey(void* key) const {
    if (num_elems_ == 0) {
      return Status::NotFound("no keys in data block");
    }

    *reinterpret_cast<CppType*>(key) = first_elem_;
    return Status::OK();
  }

  Slice Finish(rowid_t ordinal_pos) {
    if (cur_mblock_.num_elems != 0) {
      uint64_t n = cur_mblock_.max_delta - cur_mblock_.min_delta;
      cur_mblock_.bit_width = n == 0 ? 0 : Bits::Log2Ceiling64(n);
      mblocks_.push_back(cur_mblock_);
    }
    size_t num_mblocks = mblocks_.size();

    // ordinal_pos (uint32_t)
    bit_writer_.PutVlqInt(ordinal_pos);
    // Total value count (uint32_t)
    bit_writer_.PutVlqInt(num_elems_);
    // The number of miniblocks (uint32_t)
    bit_writer_.PutVlqInt(num_mblocks);

    // The bitwidth of each block is stored as a byte.
    for (size_t i = 0; i < num_mblocks; i++) {
      bit_writer_.PutValue(mblocks_[i].bit_width, kBitsPerByte);
    }
    // The frame of reference (minimum of the deltas).
    for (size_t i = 0; i < num_mblocks; i++) {
      bit_writer_.PutValue(mblocks_[i].min_delta, size_of_type * kBitsPerByte);
    }
    // Store the first value of every miniblock. So for random access, we don't need to
    // decode the deltas from the starting address of the block.
    for (size_t i = 0; i < num_mblocks; i++) {
      bit_writer_.PutValue(mblocks_[i].first_elem, size_of_type * kBitsPerByte);
    }

    // Each block is made of miniblocks, each of them binary packed with its own bit width.
    size_t k = 0;
    for (size_t i = 0; i < num_mblocks; i++) {
      for (size_t j = 0; j < mblocks_[i].num_elems - 1; j++) {
        bit_writer_.PutValue(
            deltas_[k++] - mblocks_[i].min_delta, mblocks_[i].bit_width);
      }
    }

    // Flush all buffered values. Set 'align' flag to reset buffered values.
    bit_writer_.Flush(/* align */true);
    return Slice(buffer_.data(), buffer_.size());
  }

 private:
  typedef typename TypeTraits<Type>::cpp_type CppType;
  typedef typename boost::make_unsigned<CppType>::type UnsignedCppType;

  enum {
    size_of_type = TypeTraits<Type>::size
  };

  uint64_t EstimateEncodedSize() const {
    return size_estimate_;
  }

  uint32_t num_elems_;
  vector<UnsignedCppType> deltas_;

  CppType first_elem_;
  CppType last_elem_;

  faststring buffer_;
  BitWriter bit_writer_;

  const WriterOptions* options_;
  uint64_t size_estimate_;

  struct DeltaMiniBlockState {
    // The number of elems in current miniblock. (also including the first_elem)
    size_t num_elems;
    // The minimum delta value in current miniblock
    UnsignedCppType min_delta;
    // The maximum delta value in current miniblock
    UnsignedCppType max_delta;
    // The first value in current miniblock
    CppType first_elem;
    uint8_t bit_width;
  };

  DeltaMiniBlockState cur_mblock_;
  std::vector<DeltaMiniBlockState> mblocks_;

  // kOverheadPerMiniBlock: metadata overhead for a miniblock
  // Metadata for each miniblock includes
  // - The bit width (uint8_t)
  // - The minimum delta value in the miniblock
  // - The first value in the miniblock
  static const size_t kOverheadPerMiniBlock = 2 * size_of_type + 1;
  // The number of bytes reserved for header section
  static const size_t kHeaderSizeBytes = sizeof(uint32_t) * 3;
  // The number of deltas stored in one miniblock 
  static const size_t kDeltaEntriesPerMiniBlock = 128;
  // The number of elements encoded in one miniblock 
  static const size_t kEntriesPerMiniBlock = kDeltaEntriesPerMiniBlock + 1;
  static const size_t kBitsPerByte = 8;
};

////////////////////////////////////////////////////////////
// Decoder
////////////////////////////////////////////////////////////
template<DataType Type>
class DeltaBlockDecoder : public BlockDecoder {
 public:
  explicit DeltaBlockDecoder(const Slice& slice)
  : bit_reader_(slice.data(), slice.size()),
    cur_idx_(0),
    cur_mblock_(0),
    parsed_(false) {
  }

  Status ParseHeader() {
    if (!bit_reader_.GetVlqInt(
            reinterpret_cast<int32_t*>(&ordinal_pos_base_)) ||
        !bit_reader_.GetVlqInt(
            reinterpret_cast<int32_t*>(&num_elems_)) ||
        !bit_reader_.GetVlqInt(
            reinterpret_cast<int32_t*>(&num_mblocks_))) { 
      return Status::Corruption("Header corruption");
    }

    if (!bit_reader_.GetNextBytePtr(num_mblocks_, &bit_widths_) ||
        !bit_reader_.GetNextBytePtr(
            num_mblocks_ * size_of_type,
            reinterpret_cast<const uint8_t**>(&min_deltas_)) ||
        !bit_reader_.GetNextBytePtr(
            num_mblocks_ * size_of_type,
            reinterpret_cast<const uint8_t**>(&first_values_))) {
      return Status::Corruption("Meta-data section corruption");
    }

    data_bit_offset_ = bit_reader_.position();
    parsed_ = true;
    SeekToStart();
    return Status::OK();
  }

  void SeekToStart() {
    SeekToPositionInBlock(0);
  }

  void SeekToPositionInBlock(uint pos) {
    DCHECK(parsed_) << "Must call ParseHeader()";
    if (cur_idx_ == pos) return;

    pending_.clear();
    cur_mblock_ = 0;
    bit_reader_.SeekToBit(data_bit_offset_);

    pos = std::min(num_elems_, pos);
    cur_idx_ = pos;

    // Iterate through the miniblocks until we find the one that contains the 'pos' element.
    while (pos >= kEntriesPerMiniBlock) {
      uint8_t bit_width = bit_widths_[cur_mblock_];
      int bit_offset = bit_reader_.position() + bit_width * kDeltaEntriesPerMiniBlock;
      bit_reader_.SeekToBit(bit_offset);
      pos -= kEntriesPerMiniBlock;
      cur_mblock_++;
    }
  }

  Status SeekAtOrAfterValue(const void* value_void,
                            bool* exact_match) {
    SeekToPositionInBlock(0);
    CppType target = *reinterpret_cast<const CppType*>(value_void);

    // Iterate through the miniblocks until we find the one whose subsequent miniblock doesn't
    // contain elements equal or smaller than target.
    while ((cur_idx_ + kEntriesPerMiniBlock) < num_elems_) {
      if (target >= first_values_[cur_mblock_ + 1]) {
        int last_pos = bit_reader_.position();
        uint8_t bit_width = bit_widths_[cur_mblock_];
        int bit_offset = last_pos + bit_width * kDeltaEntriesPerMiniBlock;
        bit_reader_.SeekToBit(bit_offset);
        cur_idx_ += kEntriesPerMiniBlock;
        cur_mblock_++;
      } else {
        // Stop here if the target < the first elem of the next miniblock.
        break;
      }
    }

    size_t n = kEntriesPerMiniBlock;
    n = std::min(num_elems_ - cur_idx_, n);
    if (n > 0) {
      pending_.resize(kEntriesPerMiniBlock);
      RETURN_NOT_OK(DoGetNextMiniBlock(&pending_[0]));
      // Iterate the elements in this miniblock until we find the element that is equal or greater
      // than the target.
      for (size_t i = 0; i < n; i++) {
        if (pending_[i] >= target) {
          *exact_match = pending_[i] == target;
          cur_idx_ += i;
          return Status::OK();
        }
      }
      pending_.clear();
      cur_idx_ += n;
    }

    *exact_match = false;
    if (cur_idx_ == num_elems_)
      // If the target wasn't in the block, and this was the last block, mark as not found
      return Status::NotFound("not in block");

    return Status::OK();
  }

  Status CopyNextValues(size_t* n, ColumnDataView* dst) {
    DCHECK_EQ(dst->stride(), sizeof(CppType));
    return DoGetNextValues(n, reinterpret_cast<uint8_t*>(dst->data()));
  }

  Status CopyNextValuesToArray(size_t* n, uint8_t* array) {
    return DoGetNextValues(n, array);
  }

  size_t GetCurrentIndex() const OVERRIDE {
    DCHECK(parsed_) << "Must parse header first";
    return cur_idx_;
  }

  virtual rowid_t GetFirstRowId() const OVERRIDE {
    return ordinal_pos_base_;
  }

  size_t Count() const OVERRIDE {
    return num_elems_;
  }

  bool HasNext() const OVERRIDE {
    return (num_elems_ - cur_idx_) > 0;
  }

 private:
  // Read '*n_param' elements starting from cur_idx_ and store them to 'vals_void'
  Status DoGetNextValues(size_t* n_param, uint8_t* vals_void) {
    CppType* vals = reinterpret_cast<CppType*>(vals_void);

    size_t n = *n_param;
    int start_idx = cur_idx_;
    size_t k = start_idx % kEntriesPerMiniBlock;

    n = std::min(num_elems_ - cur_idx_, n);
    if (n == 0) goto ret;

    // We always try to decode the entire miniblock at once. So if start index is not 
    // mini-block aligned, decode the values into pending list first.
    if (k > 0) {
      if (pending_.size() == 0) {
        pending_.resize(kEntriesPerMiniBlock);
        RETURN_NOT_OK(DoGetNextMiniBlock(&pending_[0]));
      }
    }

    if (pending_.size()) {
      size_t num_in_pending = std::min(n, pending_.size() - k);
      memcpy(vals, &pending_[k], num_in_pending * size_of_type);

      vals += num_in_pending;
      cur_idx_ += num_in_pending;
      n -= num_in_pending;
      if (cur_idx_ % kEntriesPerMiniBlock == 0) {
        pending_.clear();
      }
    }

    while (n >= kEntriesPerMiniBlock) {
      RETURN_NOT_OK(DoGetNextMiniBlock(reinterpret_cast<CppType*>(vals)));
      vals += kEntriesPerMiniBlock;
      cur_idx_ += kEntriesPerMiniBlock;
      n -= kEntriesPerMiniBlock;
    }

    if (n == 0) goto ret;
    // If end index is not mini-block aligned, decode the values into pending list first.
    pending_.resize(kEntriesPerMiniBlock);
    RETURN_NOT_OK(DoGetNextMiniBlock(&pending_[0]));
    memcpy(vals, &pending_[0], n * size_of_type);
    vals += n;
    cur_idx_ += n;

   ret:
    *n_param = cur_idx_ - start_idx;
    return Status::OK();
  }

  typedef typename TypeTraits<Type>::cpp_type CppType;
  typedef typename boost::make_unsigned<CppType>::type UnsignedCppType;
  enum {
    size_of_type = TypeTraits<Type>::size
  };

  Status DoGetNextMiniBlock(CppType* vals) {
    CHECK_LE(cur_mblock_, num_mblocks_);
    size_t n = kEntriesPerMiniBlock;
    n = std::min(num_elems_ - cur_idx_, n);
    // The first value in a miniblock is stored in first_values_ meta section.
    CppType elem = first_values_[cur_mblock_];
    *vals++ = elem;

    // Cast to the unsigned type first, so that integer overflow/underflow is defined.
    UnsignedCppType k = static_cast<UnsignedCppType>(elem); 
    for (size_t i = 0; i < n - 1; i++) {
      if (!bit_reader_.GetValue(bit_widths_[cur_mblock_], vals)) {
        return Status::Corruption("Bitwidth table corruption");
      }
      k += static_cast<UnsignedCppType>(*vals) + min_deltas_[cur_mblock_];
      *vals++ = static_cast<CppType>(k);
    }
    cur_mblock_++;
    return Status::OK();
  }

  rowid_t ordinal_pos_base_;
  uint32_t num_elems_;
  uint32_t num_mblocks_;
  CppType first_elem_;

  // Pointer to the list of bitwidths of miniblocks
  const uint8_t* bit_widths_;
  // Pointer to the list of min deltas of miniblocks
  const UnsignedCppType* min_deltas_;
  // Pointer to the list of first values of miniblocks
  const CppType* first_values_;

  BitReader bit_reader_;
  // Bit offset of data section.
  int data_bit_offset_;

  size_t cur_idx_;
  size_t cur_mblock_;
  bool parsed_;

  // Items that have been decoded but not yet yielded to the user.
  std::vector<CppType> pending_;
  static const size_t kDeltaEntriesPerMiniBlock = 128;
  static const size_t kEntriesPerMiniBlock = kDeltaEntriesPerMiniBlock + 1;
};

} // namespace cfile
} // namespace kudu

#endif  // KUDU_CFILE_DELTA_BLOCK_H
