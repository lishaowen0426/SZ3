#ifndef SZ3_LORENZO_PREDICTOR_HPP
#define SZ3_LORENZO_PREDICTOR_HPP

#include "SZ3/predictor/Predictor.hpp"

namespace SZ3 {

// N-dimension L-layer lorenzo predictor
template <class T, uint N, uint L>
class LorenzoPredictor : public concepts::PredictorInterface<T, N> {
   public:
    static const uint8_t predictor_id = 0b00000001;
    using block_iter = typename block_data<T, N>::block_iterator;
    bool useGrandparentPredictor = false;
    double grandparentPredictorRatio = 0.9;
    const int64_t *predIdx = nullptr;
    size_t predIdxSize = 0;
    const int64_t *anchorIdx = nullptr;
    size_t anchorIdxSize = 0;
    const double *anchorValue = nullptr;
    size_t anchorValueSize = 0;

    LorenzoPredictor() { this->noise = 0; }

    LorenzoPredictor(double eb) {
        this->noise = 0;
        if (L == 1) {
            if (N == 1) {
                this->noise = 0.5 * eb;
            } else if (N == 2) {
                this->noise = 0.81 * eb;
            } else if (N == 3) {
                this->noise = 1.22 * eb;
            } else if (N == 4) {
                this->noise = 1.79 * eb;
            }
        } else if (L == 2) {
            if (N == 1) {
                this->noise = 1.08 * eb;
            } else if (N == 2) {
                this->noise = 2.76 * eb;
            } else if (N == 3) {
                this->noise = 6.8 * eb;
            }
        }
    }

    bool precompress(const block_iter &) override { return true; }

    void precompress_block_commit() noexcept override {}

    bool predecompress(const block_iter &) override { return true; }

    void save(uchar *&c) override {}

    void load(const uchar *&c, size_t &remaining_length) override {}

    void print() const override {
        std::cout << L << "-Layer " << N << "D Lorenzo predictor, noise = " << noise << "\n";
    }

    size_t get_padding() override { return 2; }

    ALWAYS_INLINE T estimate_error(const block_iter &block, T *d, const std::array<size_t, N> &index) override {
        return fabs(*d - predict(block, d, index)) + this->noise;
    }

    ALWAYS_INLINE T predict(const block_iter &block, T *d, const std::array<size_t, N> &index) override {
        auto ds = block.get_dim_strides();
        if constexpr (N == 1 && L == 1) {
            if (predIdx != nullptr) {
                auto block_range = block.get_block_range();
                const size_t current_index = block_range[0].first + index[0];
                if (current_index >= predIdxSize) {
                    throw std::runtime_error("predIdxSize is smaller than the 1D data extent");
                }
                auto lookup_anchor_value = [&](size_t target_index, T &value) -> bool {
                    if (anchorIdx != nullptr || anchorValue != nullptr) {
                        if (anchorIdx == nullptr || anchorValue == nullptr || anchorIdxSize != anchorValueSize) {
                            throw std::runtime_error("anchor arrays are not configured consistently");
                        }
                        for (size_t anchor_pos = 0; anchor_pos < anchorIdxSize; anchor_pos++) {
                            if (anchorIdx[anchor_pos] == static_cast<int64_t>(target_index)) {
                                value = static_cast<T>(anchorValue[anchor_pos]);
                                return true;
                            }
                        }
                    }
                    return false;
                };
                auto lookup_stream_value = [&](size_t target_index) -> T {
                    T value;
                    if (lookup_anchor_value(target_index, value)) {
                        return value;
                    }
                    return *(d + (static_cast<int64_t>(target_index) - static_cast<int64_t>(current_index)));
                };
                const int64_t previous_index = predIdx[current_index];
                if (previous_index < 0) {
                    T anchor_value;
                    if (lookup_anchor_value(current_index, anchor_value)) {
                        return anchor_value;
                    }
                    return T(0);
                }
                const T parent_value = lookup_stream_value(static_cast<size_t>(previous_index));
                if (!useGrandparentPredictor) {
                    return parent_value;
                }
                if (static_cast<size_t>(previous_index) >= predIdxSize) {
                    throw std::runtime_error("predIdx contains an out-of-range parent index");
                }
                const int64_t grandparent_index = predIdx[static_cast<size_t>(previous_index)];
                if (grandparent_index < 0) {
                    return parent_value;
                }
                const T grandparent_value = lookup_stream_value(static_cast<size_t>(grandparent_index));
                const T parent_ratio = static_cast<T>(grandparentPredictorRatio);
                const T grandparent_ratio = static_cast<T>(1.0 - grandparentPredictorRatio);
                return parent_ratio * parent_value + grandparent_ratio * grandparent_value;
            }
            return prev1(d, 1);
        } else if constexpr (N == 2 && L == 1) {
            return prev2(d, ds, 0, 1) + prev2(d, ds, 1, 0) - prev2(d, ds, 1, 1);
        } else if constexpr (N == 3 && L == 1) {
            return prev3(d, ds, 0, 0, 1) + prev3(d, ds, 0, 1, 0) + prev3(d, ds, 1, 0, 0) - prev3(d, ds, 0, 1, 1) -
                   prev3(d, ds, 1, 0, 1) - prev3(d, ds, 1, 1, 0) + prev3(d, ds, 1, 1, 1);
        } else if constexpr (N == 4 && L == 1) {
            return prev4(d, ds, 0, 0, 0, 1) + prev4(d, ds, 0, 0, 1, 0) - prev4(d, ds, 0, 0, 1, 1) +
                   prev4(d, ds, 0, 1, 0, 0) - prev4(d, ds, 0, 1, 0, 1) - prev4(d, ds, 0, 1, 1, 0) +
                   prev4(d, ds, 0, 1, 1, 1) + prev4(d, ds, 1, 0, 0, 0) - prev4(d, ds, 1, 0, 0, 1) -
                   prev4(d, ds, 1, 0, 1, 0) + prev4(d, ds, 1, 0, 1, 1) - prev4(d, ds, 1, 1, 0, 0) +
                   prev4(d, ds, 1, 1, 0, 1) + prev4(d, ds, 1, 1, 1, 0) - prev4(d, ds, 1, 1, 1, 1);
        } else if constexpr (N == 1 && L == 2) {
            return 2 * prev1(d, 1) - prev1(d, 2);
        } else if constexpr (N == 2 && L == 2) {
            return 2 * prev2(d, ds, 0, 1) - prev2(d, ds, 0, 2) + 2 * prev2(d, ds, 1, 0) - 4 * prev2(d, ds, 1, 1) +
                   2 * prev2(d, ds, 1, 2) - prev2(d, ds, 2, 0) + 2 * prev2(d, ds, 2, 1) - prev2(d, ds, 2, 2);
        } else if constexpr (N == 3 && L == 2) {
            return 2 * prev3(d, ds, 0, 0, 1) - prev3(d, ds, 0, 0, 2) + 2 * prev3(d, ds, 0, 1, 0) -
                   4 * prev3(d, ds, 0, 1, 1) + 2 * prev3(d, ds, 0, 1, 2) - prev3(d, ds, 0, 2, 0) +
                   2 * prev3(d, ds, 0, 2, 1) - prev3(d, ds, 0, 2, 2) + 2 * prev3(d, ds, 1, 0, 0) -
                   4 * prev3(d, ds, 1, 0, 1) + 2 * prev3(d, ds, 1, 0, 2) - 4 * prev3(d, ds, 1, 1, 0) +
                   8 * prev3(d, ds, 1, 1, 1) - 4 * prev3(d, ds, 1, 1, 2) + 2 * prev3(d, ds, 1, 2, 0) -
                   4 * prev3(d, ds, 1, 2, 1) + 2 * prev3(d, ds, 1, 2, 2) - prev3(d, ds, 2, 0, 0) +
                   2 * prev3(d, ds, 2, 0, 1) - prev3(d, ds, 2, 0, 2) + 2 * prev3(d, ds, 2, 1, 0) -
                   4 * prev3(d, ds, 2, 1, 1) + 2 * prev3(d, ds, 2, 1, 2) - prev3(d, ds, 2, 2, 0) +
                   2 * prev3(d, ds, 2, 2, 1) - prev3(d, ds, 2, 2, 2);
        } else {
            static_assert(N <= 4 && L <= 2, "Unsupported dimension or layer configuration");
            return T(0);
        }
    }

   protected:
    T noise = 0;

   private:
    // Helper functions for Lorenzo prediction
    ALWAYS_INLINE T prev1(T *d, size_t i) { return *(d - i); }
    ALWAYS_INLINE T prev2(T *d, const std::array<size_t, N> &ds, size_t j, size_t i) { return *(d - (j * ds[0] + i)); }
    ALWAYS_INLINE T prev3(T *d, const std::array<size_t, N> &ds, size_t k, size_t j, size_t i) {
        return *(d - (k * ds[1] + j * ds[0] + i));
    }
    ALWAYS_INLINE T prev4(T *d, const std::array<size_t, N> &ds, size_t t, size_t k, size_t j, size_t i) {
        return *(d - (t * ds[2] + k * ds[1] + j * ds[0] + i));
    }
};
}  // namespace SZ3
#endif
