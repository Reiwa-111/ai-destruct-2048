#include <algorithm>
#include <array>
#include <cstdint>

namespace {
constexpr int kBoardCells = 16;
constexpr int kPatternCount = 4;
constexpr int kTupleSize = 6;
constexpr int kSymmetryCount = 8;
constexpr int kCellStates = 16;
constexpr int kTableSize = kCellStates * kCellStates * kCellStates *
                           kCellStates * kCellStates * kCellStates;

constexpr int kPatterns[kPatternCount][kTupleSize] = {
    {0, 1, 2, 3, 4, 5},
    {4, 5, 6, 7, 8, 9},
    {0, 1, 4, 5, 8, 9},
    {0, 1, 5, 6, 9, 10},
};

using Board = std::array<std::uint64_t, kBoardCells>;

inline std::uint8_t rank_of(std::uint64_t tile) {
    if (tile == 0) return 0;
    const auto rank = static_cast<unsigned>(__builtin_ctzll(tile));
    return static_cast<std::uint8_t>(std::min(rank, 15U));
}

Board reflect_horizontal(const Board& board) {
    Board result{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            result[row * 4 + col] = board[row * 4 + (3 - col)];
        }
    }
    return result;
}

Board rotate90(const Board& board) {
    Board result{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            result[row * 4 + col] = board[(3 - col) * 4 + row];
        }
    }
    return result;
}

template <bool Update>
double evaluate_or_update(const std::uint64_t* raw_board, float* tables,
                          double target, double alpha) {
    Board original{};
    std::copy(raw_board, raw_board + kBoardCells, original.begin());

    double total = 0.0;
    for (int reflected = 0; reflected < 2; ++reflected) {
        Board transformed = reflected ? reflect_horizontal(original) : original;
        for (int rotation = 0; rotation < 4; ++rotation) {
            std::array<std::uint8_t, kBoardCells> ranks{};
            for (int i = 0; i < kBoardCells; ++i) ranks[i] = rank_of(transformed[i]);
            for (int pattern = 0; pattern < kPatternCount; ++pattern) {
                std::uint32_t index = 0;
                for (int cell = 0; cell < kTupleSize; ++cell) {
                    index = index * kCellStates + ranks[kPatterns[pattern][cell]];
                }
                total += tables[pattern * kTableSize + index];
            }
            transformed = rotate90(transformed);
        }
    }

    const double prediction = total / 8.0;
    if constexpr (Update) {
        const double error = target - prediction;
        // The prediction is averaged over all eight D4 symmetries. Each
        // active table feature therefore has derivative 1 / kSymmetryCount.
        const float increment = static_cast<float>(alpha * error / kSymmetryCount);
        for (int reflected = 0; reflected < 2; ++reflected) {
            Board transformed = reflected ? reflect_horizontal(original) : original;
            for (int rotation = 0; rotation < 4; ++rotation) {
                std::array<std::uint8_t, kBoardCells> ranks{};
                for (int i = 0; i < kBoardCells; ++i) ranks[i] = rank_of(transformed[i]);
            for (int pattern = 0; pattern < kPatternCount; ++pattern) {
                    std::uint32_t index = 0;
                    for (int cell = 0; cell < kTupleSize; ++cell) {
                        index = index * kCellStates + ranks[kPatterns[pattern][cell]];
                    }
                    tables[pattern * kTableSize + index] += increment;
                }
                transformed = rotate90(transformed);
            }
        }
        return error;
    }
    return prediction;
}
}  // namespace

extern "C" double destroyer_ntuple_value(const std::uint64_t* board, const float* tables) {
    return evaluate_or_update<false>(board, const_cast<float*>(tables), 0.0, 0.0);
}

extern "C" double destroyer_ntuple_update(const std::uint64_t* board, float* tables,
                                           double target, double alpha) {
    return evaluate_or_update<true>(board, tables, target, alpha);
}
