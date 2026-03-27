#ifndef SZ3_BFS_BLOCKWISE_DECOMPOSITION_HPP
#define SZ3_BFS_BLOCKWISE_DECOMPOSITION_HPP

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <type_traits>

#include "Decomposition.hpp"
#include "SZ3/def.hpp"
#include "SZ3/predictor/LorenzoPredictor.hpp"
#include "SZ3/predictor/Predictor.hpp"
#include "SZ3/quantizer/LinearQuantizer.hpp"
#include "SZ3/utils/BFSBlockwiseIterator.hpp"
#include "SZ3/utils/Config.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "SZ3/utils/Timer.hpp"

namespace SZ3 {
template <class T, uint N, class Predictor, class Quantizer>
class BFSBlockwiseDecomposition : public concepts::DecompositionInterface<T, int, N> {
   public:
    using Block_iter = typename bfs_block_data<T, N>::block_iterator;

    BFSBlockwiseDecomposition(const Config &conf, Predictor predictor, Quantizer quantizer)
        : predictor(predictor), quantizer(quantizer), fallback_predictor(conf.absErrorBound) {
        static_assert(std::is_base_of<concepts::PredictorInterface<T, N>, Predictor>::value,
                      "must implement the Predictor interface");
    }

    std::vector<int> compress(const Config &conf, T *data) override {
        if constexpr (!(std::is_same<Predictor, LorenzoPredictor<T, N, 1>>::value ||
                        std::is_same<Predictor, LorenzoPredictor<T, N, 2>>::value)) {
            throw std::invalid_argument("BFSBlockwiseDecomposition currently requires a Lorenzo predictor");
        }

        auto data_with_padding = std::make_shared<bfs_block_data<T, N>>(data, conf.dims, predictor.get_padding(), true);
        auto block = conf.blockSizes.empty() ? data_with_padding->block_iter(conf.blockSize)
                                             : data_with_padding->block_iter(conf.blockSizes);
        std::ofstream quant_inds_stream;
        const bool dump_quant_inds = !conf.quantIndsPath.empty();
        bool wrote_block = false;
        if (dump_quant_inds) {
            quant_inds_stream.open(conf.quantIndsPath, std::ios::out | std::ios::trunc);
            if (!quant_inds_stream) {
                throw std::runtime_error("failed to open quantization-index output file: " + conf.quantIndsPath);
            }
        }
        std::vector<int> quant_inds;
        quant_inds.reserve(conf.num);
        do {
            const size_t block_begin = quant_inds.size();
            if (!predictor.precompress(block)) {
                fallback_predictor.precompress(block);
                fallback_predictor.precompress_block_commit();
                {
                    auto block_range = block.get_block_range();
                    std::cout << "bfs_block_size=";
                    for (uint dim = 0; dim < N; dim++) {
                        if (dim) std::cout << "x";
                        std::cout << (block_range[dim].second - block_range[dim].first);
                    }
                    std::cout << "\n";
                }
                Block_iter::foreach (block, [&](T *c, const std::array<size_t, N> &index) {
                    T pred = fallback_predictor.predict(block, c, index);
                    quant_inds.push_back(quantizer.quantize_and_overwrite(*c, pred));
                });
            } else {
                predictor.precompress_block_commit();
                {
                    auto block_range = block.get_block_range();
                    std::cout << "bfs_block_size=";
                    for (uint dim = 0; dim < N; dim++) {
                        if (dim) std::cout << "x";
                        std::cout << (block_range[dim].second - block_range[dim].first);
                    }
                    std::cout << "\n";
                }
                Block_iter::foreach (block, [&](T *c, const std::array<size_t, N> &index) {
                    T pred = predictor.predict(block, c, index);
                    quant_inds.push_back(quantizer.quantize_and_overwrite(*c, pred));
                });
            }
            if (dump_quant_inds) {
                if (wrote_block) {
                    quant_inds_stream << '\n';
                }
                for (size_t i = block_begin; i < quant_inds.size(); i++) {
                    if (i > block_begin) {
                        quant_inds_stream << ' ';
                    }
                    quant_inds_stream << quant_inds[i];
                }
                wrote_block = true;
            }
        } while (block.next());
        return quant_inds;
    }

    T *decompress(const Config &conf, std::vector<int> &quant_inds, T *dec_data) override {
        if constexpr (!(std::is_same<Predictor, LorenzoPredictor<T, N, 1>>::value ||
                        std::is_same<Predictor, LorenzoPredictor<T, N, 2>>::value)) {
            throw std::invalid_argument("BFSBlockwiseDecomposition currently requires a Lorenzo predictor");
        }

        int *quant_inds_pos = &quant_inds[0];
        auto data_with_padding =
            std::make_shared<bfs_block_data<T, N>>(dec_data, conf.dims, predictor.get_padding(), false);
        auto block = conf.blockSizes.empty() ? data_with_padding->block_iter(conf.blockSize)
                                             : data_with_padding->block_iter(conf.blockSizes);
        do {
            if (!predictor.predecompress(block)) {
                fallback_predictor.predecompress(block);
                Block_iter::foreach (block, [&](T *c, const std::array<size_t, N> &index) {
                    T pred = fallback_predictor.predict(block, c, index);
                    *c = quantizer.recover(pred, *(quant_inds_pos++));
                });
            } else {
                Block_iter::foreach (block, [&](T *c, const std::array<size_t, N> &index) {
                    T pred = predictor.predict(block, c, index);
                    *c = quantizer.recover(pred, *(quant_inds_pos++));
                });
            }
        } while (block.next());

        return dec_data;
    }

    void save(uchar *&c) override {
        fallback_predictor.save(c);
        predictor.save(c);
        quantizer.save(c);
    }

    void load(const uchar *&c, size_t &remaining_length) override {
        fallback_predictor.load(c, remaining_length);
        predictor.load(c, remaining_length);
        quantizer.load(c, remaining_length);
    }

    std::pair<int, int> get_out_range() override { return quantizer.get_out_range(); }

   private:
    Predictor predictor;
    Quantizer quantizer;
    LorenzoPredictor<T, N, 1> fallback_predictor;
};

template <class T, uint N, class Predictor, class Quantizer>
BFSBlockwiseDecomposition<T, N, Predictor, Quantizer> make_decomposition_bfs_blockwise(const Config &conf,
                                                                                        Predictor predictor,
                                                                                        Quantizer quantizer) {
    return BFSBlockwiseDecomposition<T, N, Predictor, Quantizer>(conf, predictor, quantizer);
}

}  // namespace SZ3
#endif
