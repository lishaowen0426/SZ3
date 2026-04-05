#ifndef SZ3_SPLINE_INTERPOLATION_DECOMPOSITION_HPP
#define SZ3_SPLINE_INTERPOLATION_DECOMPOSITION_HPP

#include <cassert>
#include <cmath>
#include <vector>

#include "Decomposition.hpp"
#include "SZ3/def.hpp"
#include "SZ3/quantizer/Quantizer.hpp"
#include "SZ3/utils/BlockwiseIterator.hpp"
#include "SZ3/utils/Config.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "SZ3/utils/MemoryUtil.hpp"

namespace SZ3 {

/**
 * 1D-only blockwise spline interpolation decomposition.
 *
 * The array is split into contiguous blocks (size conf.blockSize, or the whole
 * array if blockSize == 0). Each block is compressed independently:
 *   - block[0] is quantized with pred=0
 *   - remaining elements use multilevel natural cubic spline prediction
 *
 * This mirrors quantize_with_spline_1d() in predict_and_quant.py, applied
 * per block to keep working sets cache-friendly.
 */
template <class T, class Quantizer>
class SplineInterpolationDecomposition : public concepts::DecompositionInterface<T, int, 1> {
   public:
    static constexpr uint N = 1;

    SplineInterpolationDecomposition(const Config &conf, Quantizer quantizer)
        : quantizer(quantizer) {
        static_assert(std::is_base_of<concepts::QuantizerInterface<T, int>, Quantizer>::value,
                      "must implement the quantizer interface");
        std::copy_n(conf.dims.begin(), 1, original_dimensions.begin());
    }

    std::vector<int> compress(const Config &conf, T *data) override {
        std::copy_n(conf.dims.begin(), 1, original_dimensions.begin());
        size_t n = original_dimensions[0];
        size_t bsz = !conf.blockSizes.empty() ? conf.blockSizes[0]
                     : (conf.blockSize > 0)   ? (size_t)conf.blockSize
                                              : n;

        std::vector<int> out;
        out.reserve(n);

        auto bd = std::make_shared<block_data<T, 1>>(
            data, std::vector<size_t>{n}, 0, false);
        auto blk = bd->block_iter(bsz);
        do {
            auto range = blk.get_block_range();
            T *p = blk.get_block_data(0);
            size_t len = range[0].second - range[0].first;
            compress_block(p, len, out);
        } while (blk.next());

        quantizer.postcompress_data();
        return out;
    }

    T *decompress(const Config &conf, std::vector<int> &quant_inds_in, T *dec_data) override {
        std::copy_n(conf.dims.begin(), 1, original_dimensions.begin());
        size_t n = original_dimensions[0];
        size_t bsz = !conf.blockSizes.empty() ? conf.blockSizes[0]
                     : (conf.blockSize > 0)   ? (size_t)conf.blockSize
                                              : n;

        const int *qi = quant_inds_in.data();
        auto bd = std::make_shared<block_data<T, 1>>(
            dec_data, std::vector<size_t>{n}, 0, false);
        auto blk = bd->block_iter(bsz);
        do {
            auto range = blk.get_block_range();
            T *p = blk.get_block_data(0);
            size_t len = range[0].second - range[0].first;
            qi = decompress_block(p, len, qi);
        } while (blk.next());

        quantizer.postdecompress_data();
        return dec_data;
    }

    void save(uchar *&c) override {
        write(original_dimensions.data(), 1, c);
        quantizer.save(c);
    }

    void load(const uchar *&c, size_t &remaining_length) override {
        read(original_dimensions.data(), 1, c, remaining_length);
        quantizer.load(c, remaining_length);
    }

    std::pair<int, int> get_out_range() override { return quantizer.get_out_range(); }

   private:
    void compress_block(T *data, size_t n, std::vector<int> &out) {
        out.push_back(quantizer.quantize_and_overwrite(data[0], T(0)));
        int levels = static_cast<int>(std::ceil(std::log2(std::max((size_t)2, n))));
        for (int level = levels; level > 0; --level) {
            size_t stride = size_t(1) << (level - 1);
            spline_predict_1d(data, 0, n - 1, stride,
                [&](size_t, T &d, T pred) {
                    out.push_back(quantizer.quantize_and_overwrite(d, pred));
                });
        }
    }

    const int *decompress_block(T *data, size_t n, const int *qi) {
        data[0] = quantizer.recover(T(0), *qi++);
        int levels = static_cast<int>(std::ceil(std::log2(std::max((size_t)2, n))));
        for (int level = levels; level > 0; --level) {
            size_t stride = size_t(1) << (level - 1);
            spline_predict_1d(data, 0, n - 1, stride,
                [&](size_t, T &d, T pred) {
                    d = quantizer.recover(pred, *qi++);
                });
        }
        return qi;
    }

    /**
     * Solve a natural cubic spline through points (xs[i], ys[i]) and evaluate
     * at query positions xq[j], writing results to out[j].
     * Uses Thomas algorithm for the tridiagonal system.
     */
    static void natural_cubic_spline_eval(const std::vector<double> &xs,
                                          const std::vector<double> &ys,
                                          const std::vector<double> &xq,
                                          std::vector<double> &out) {
        int m = (int)xs.size();
        out.resize(xq.size());

        if (m == 1) {
            std::fill(out.begin(), out.end(), ys[0]);
            return;
        }
        if (m == 2) {
            double slope = (ys[1] - ys[0]) / (xs[1] - xs[0]);
            for (size_t j = 0; j < xq.size(); j++)
                out[j] = ys[0] + slope * (xq[j] - xs[0]);
            return;
        }

        std::vector<double> h(m - 1);
        for (int i = 0; i < m - 1; i++) h[i] = xs[i + 1] - xs[i];

        std::vector<double> rhs(m, 0.0);
        for (int i = 1; i < m - 1; i++)
            rhs[i] = 3.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]);

        // Thomas algorithm for natural spline (M[0]=M[m-1]=0)
        std::vector<double> M(m, 0.0);
        std::vector<double> c(m - 1), d(m - 1);
        c[0] = 0.0;
        d[0] = rhs[1] / (2.0 * (h[0] + h[1]));
        for (int i = 1; i < m - 2; i++) {
            double denom = 2.0 * (h[i] + h[i + 1]) - h[i] * c[i - 1];
            c[i] = h[i + 1] / denom;
            d[i] = (rhs[i + 1] - h[i] * d[i - 1]) / denom;
        }
        M[m - 2] = d[m - 3];
        for (int i = m - 3; i >= 1; i--)
            M[i] = d[i - 1] - c[i - 1] * M[i + 1];

        // knots are uniformly spaced — compute segment in O(1)
        double h0 = h[0];
        for (size_t j = 0; j < xq.size(); j++) {
            double x = xq[j];
            int seg = std::min((int)((x - xs[0]) / h0), m - 2);
            double t = x - xs[seg];
            double hi = h[seg];
            double a = ys[seg];
            double b = (ys[seg + 1] - ys[seg]) / hi - hi * (2.0 * M[seg] + M[seg + 1]) / 3.0;
            double c_ = M[seg];
            double d_ = (M[seg + 1] - M[seg]) / (3.0 * hi);
            out[j] = a + b * t + c_ * t * t + d_ * t * t * t;
        }
    }

    template <class QuantizeFunc>
    void spline_predict_1d(T *data, size_t begin, size_t end, size_t stride,
                           QuantizeFunc &&quantize_func) {
        std::vector<double> known_x, known_y;
        for (size_t pos = begin; pos <= end; pos += 2 * stride) {
            known_x.push_back((double)pos);
            known_y.push_back((double)data[pos]);
        }
        if (known_x.size() < 2) return;

        std::vector<double> unknown_x;
        std::vector<size_t> unknown_idx;
        for (size_t pos = begin + stride; pos <= end; pos += 2 * stride) {
            unknown_x.push_back((double)pos);
            unknown_idx.push_back(pos);
        }
        if (unknown_x.empty()) return;

        std::vector<double> preds;
        natural_cubic_spline_eval(known_x, known_y, unknown_x, preds);

        for (size_t j = 0; j < unknown_idx.size(); j++) {
            size_t idx = unknown_idx[j];
            quantize_func(idx, data[idx], (T)preds[j]);
        }
    }

    Quantizer quantizer;
    std::array<size_t, 1> original_dimensions{0};
};

template <class T, class Quantizer>
SplineInterpolationDecomposition<T, Quantizer> make_decomposition_spline_interpolation(
    const Config &conf, Quantizer quantizer) {
    return SplineInterpolationDecomposition<T, Quantizer>(conf, quantizer);
}

}  // namespace SZ3

#endif  // SZ3_SPLINE_INTERPOLATION_DECOMPOSITION_HPP
