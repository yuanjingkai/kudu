// Copyright (c) 2013, Cloudera, inc.
// Confidential Cloudera Information: Covered by NDA.
//
// Delta encoding is adapted from the Binary packing described in
// "Decoding billions of integers per second through vectorization"
// by D. Lemire and L. Boytsov
#include "kudu/cfile/delta_block.h"
#include "kudu/cfile/simdbitpacking.h"
#include "kudu/util/bit-stream-utils.h"
#include "kudu/util/bit-stream-utils.inline.h"

namespace kudu { namespace cfile {

using kudu::coding::usimdunpack;

// Template specialization for INT32 data type.
template<>
Status DeltaBlockDecoder<INT32>::DoGetNextMiniBlock(int32_t* vals) {
  DCHECK_LE(cur_mblock_, num_mblocks_);
  size_t n = kEntriesPerMiniBlock;
  n = std::min(num_elems_ - cur_idx_, n);

  // The first value in a miniblock is stored in
  // first_values_ meta section.
  int32_t elem = first_values_[cur_mblock_];
  *vals++ = elem;

  // Cast to the unsigned type first, so that integer
  // overflow/underflow is defined.
  uint32_t k = static_cast<uint32_t>(elem);

  uint8_t bit_width = bit_widths_[cur_mblock_];
  size_t i = 0;
  for (; i + 128 <= n - 1; i += 128) {
    const uint8_t *ptr = NULL;
    if (!bit_reader_.GetNextBytePtr(bit_width * 128 / 8, &ptr)) {
      return Status::Corruption("Bitwidth table corruption");
    }
    usimdunpack(reinterpret_cast<const __m128i *>(ptr),
                reinterpret_cast<uint32_t *>(vals),
                bit_width,
                min_deltas_[cur_mblock_]);
    for (size_t j = 0; j < 128; j++) {
      k += static_cast<uint32_t>(*vals);
      *vals++ = static_cast<int32_t>(k);
    }
  }

  for (; i < n - 1; i++) {
    if (!bit_reader_.GetValue(bit_width, vals)) {
      return Status::Corruption("Bitwidth table corruption");
    }
    k += static_cast<uint32_t>(*vals) + min_deltas_[cur_mblock_];
    *vals++ = static_cast<int32_t>(k);
  }
  cur_mblock_++;
  return Status::OK();
}

} // namespace cfile
} // namespace kudu
