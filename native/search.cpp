#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
constexpr std::uint32_t kLineCount = 1U << 20;  // Four 5-bit ranks.
constexpr int kTupleStates = 16;
constexpr int kTupleSize = 6;
constexpr int kTupleCount = 4;
constexpr int kSymmetryCount = 8;
constexpr std::uint32_t kTupleTableSize = 1U << 24;
constexpr std::size_t kCacheBucketCount = 1U << 16;
constexpr std::size_t kCacheWays = 4;
constexpr std::array<std::array<int, kTupleSize>, kTupleCount> kPatterns{{
    {{0, 1, 2, 3, 4, 5}},
    {{4, 5, 6, 7, 8, 9}},
    {{0, 1, 4, 5, 8, 9}},
    {{0, 1, 5, 6, 9, 10}},
}};

constexpr auto make_symmetry_maps() {
    std::array<std::array<std::uint8_t, 16>, kSymmetryCount> maps{};
    for (int reflected = 0; reflected < 2; ++reflected) {
        std::array<std::uint8_t, 16> current{};
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                const int source_col = reflected ? 3 - col : col;
                current[row * 4 + col] = static_cast<std::uint8_t>(row * 4 + source_col);
            }
        }
        for (int rotation = 0; rotation < 4; ++rotation) {
            maps[reflected * 4 + rotation] = current;
            std::array<std::uint8_t, 16> rotated{};
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    rotated[row * 4 + col] = current[(3 - col) * 4 + row];
                }
            }
            current = rotated;
        }
    }
    return maps;
}

constexpr auto kSymmetryMaps = make_symmetry_maps();

constexpr auto make_pattern_cell_maps() {
    std::array<
        std::array<std::array<std::uint8_t, kTupleSize>, kTupleCount>,
        kSymmetryCount
    > maps{};
    for (int symmetry = 0; symmetry < kSymmetryCount; ++symmetry) {
        for (int pattern = 0; pattern < kTupleCount; ++pattern) {
            for (int cell = 0; cell < kTupleSize; ++cell) {
                maps[symmetry][pattern][cell] =
                    kSymmetryMaps[symmetry][kPatterns[pattern][cell]];
            }
        }
    }
    return maps;
}

constexpr auto kPatternCellMaps = make_pattern_cell_maps();

struct Board {
    std::array<std::uint8_t, 16> cells{};
    bool operator==(const Board& other) const noexcept { return cells == other.cells; }
};

Board board_from_ranks(const std::uint8_t* ranks);

struct LineEntry {
    std::uint32_t result = 0;
    std::uint64_t reward = 0;
    std::uint8_t merges = 0;
};

std::vector<LineEntry> g_lines;
std::vector<float> g_heuristics;
std::once_flag g_init_flag;

std::uint32_t reverse_line(std::uint32_t key) {
    return ((key & 31U) << 15) |
           ((key & (31U << 5)) << 5) |
           ((key & (31U << 10)) >> 5) |
           ((key & (31U << 15)) >> 15);
}

std::uint32_t make_line(const std::array<std::uint8_t, 4>& values) {
    return static_cast<std::uint32_t>(values[0]) |
           (static_cast<std::uint32_t>(values[1]) << 5) |
           (static_cast<std::uint32_t>(values[2]) << 10) |
           (static_cast<std::uint32_t>(values[3]) << 15);
}

std::array<std::uint8_t, 4> decode_line(std::uint32_t key) {
    return {{
        static_cast<std::uint8_t>(key & 31U),
        static_cast<std::uint8_t>((key >> 5) & 31U),
        static_cast<std::uint8_t>((key >> 10) & 31U),
        static_cast<std::uint8_t>((key >> 15) & 31U),
    }};
}

float heuristic_line(const std::array<std::uint8_t, 4>& line) {
    float sum = 0.0f;
    int empty = 0;
    int merges = 0;
    int previous = 0;
    int merge_counter = 0;
    for (int rank : line) {
        sum += std::pow(static_cast<float>(rank), 3.5f);
        if (rank == 0) {
            ++empty;
        } else {
            if (previous == rank) {
                ++merge_counter;
            } else if (merge_counter > 0) {
                merges += 1 + merge_counter;
                merge_counter = 0;
            }
            previous = rank;
        }
    }
    if (merge_counter > 0) merges += 1 + merge_counter;

    float mono_left = 0.0f;
    float mono_right = 0.0f;
    for (int i = 1; i < 4; ++i) {
        const float previous_rank = static_cast<float>(line[i - 1]);
        const float rank = static_cast<float>(line[i]);
        const float previous_power = previous_rank * previous_rank * previous_rank * previous_rank;
        const float rank_power = rank * rank * rank * rank;
        if (previous_rank > rank) mono_left += previous_power - rank_power;
        else mono_right += rank_power - previous_power;
    }

    return 200000.0f + 270.0f * empty + 700.0f * merges -
           47.0f * std::min(mono_left, mono_right) - 11.0f * sum;
}

void initialize_tables() {
    std::call_once(g_init_flag, [] {
        g_lines.resize(kLineCount);
        g_heuristics.resize(kLineCount);
        for (std::uint32_t key = 0; key < kLineCount; ++key) {
            const auto line = decode_line(key);

            std::array<std::uint8_t, 4> compact{};
            int compact_count = 0;
            for (std::uint8_t rank : line) {
                if (rank != 0) compact[compact_count++] = rank;
            }

            std::array<std::uint8_t, 4> packed{};
            int output_count = 0;
            std::uint64_t reward = 0;
            int merges = 0;
            for (int i = 0; i < compact_count;) {
                if (i + 1 < compact_count && compact[i] == compact[i + 1]) {
                    const int new_rank = std::min<int>(compact[i] + 1, 31);
                    packed[output_count++] = static_cast<std::uint8_t>(new_rank);
                    reward += (std::uint64_t{1} << new_rank);
                    ++merges;
                    i += 2;
                } else {
                    packed[output_count++] = compact[i];
                    ++i;
                }
            }

            g_lines[key] = LineEntry{make_line(packed), reward, static_cast<std::uint8_t>(merges)};
            g_heuristics[key] = heuristic_line(line);
        }
    });
}

LineEntry move_line(std::uint32_t key, bool reverse) {
    if (!reverse) return g_lines[key];
    const LineEntry& entry = g_lines[reverse_line(key)];
    return LineEntry{reverse_line(entry.result), entry.reward, entry.merges};
}

struct MoveResult {
    Board board;
    std::uint64_t reward = 0;
    int merges = 0;
    bool moved = false;
};

struct LegalMove {
    int action = 0;
    MoveResult result{};
};

// A 4x4 board can have at most four legal moves. Keep the move list inline:
// this hot path used to allocate a std::vector at every visited search node.
struct LegalMoves {
    std::array<LegalMove, 4> entries{};
    std::size_t count = 0;

    bool empty() const noexcept { return count == 0; }
    std::size_t size() const noexcept { return count; }
    const LegalMove& front() const noexcept { return entries[0]; }
    const LegalMove* begin() const noexcept { return entries.data(); }
    const LegalMove* end() const noexcept { return entries.data() + count; }
};

MoveResult apply_move(const Board& board, int action) {
    MoveResult result;
    result.board = board;
    if (action < 0 || action > 3) return result;

    const bool vertical = action < 2;
    const bool reverse = action == 1 || action == 3;
    for (int line_index = 0; line_index < 4; ++line_index) {
        std::array<std::uint8_t, 4> line{};
        for (int offset = 0; offset < 4; ++offset) {
            const int row = vertical ? offset : line_index;
            const int col = vertical ? line_index : offset;
            line[offset] = board.cells[row * 4 + col];
        }
        const LineEntry moved = move_line(make_line(line), reverse);
        result.reward += moved.reward;
        result.merges += moved.merges;
        const auto decoded = decode_line(moved.result);
        for (int offset = 0; offset < 4; ++offset) {
            const int row = vertical ? offset : line_index;
            const int col = vertical ? line_index : offset;
            result.board.cells[row * 4 + col] = decoded[offset];
        }
    }
    result.moved = !(result.board == board);
    return result;
}

LegalMoves legal_moves(const Board& board) {
    LegalMoves moves;
    for (int action = 0; action < 4; ++action) {
        MoveResult result = apply_move(board, action);
        if (result.moved) moves.entries[moves.count++] = LegalMove{action, result};
    }
    return moves;
}

int count_empty(const Board& board) {
    return static_cast<int>(std::count(board.cells.begin(), board.cells.end(), std::uint8_t{0}));
}

float classic_heuristic(const Board& board) {
    if (count_empty(board) == 0) {
        bool merge_available = false;
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                const auto value = board.cells[row * 4 + col];
                if (row + 1 < 4 && value == board.cells[(row + 1) * 4 + col]) merge_available = true;
                if (col + 1 < 4 && value == board.cells[row * 4 + col + 1]) merge_available = true;
            }
        }
        if (!merge_available) return 0.0f;
    }

    float score = 0.0f;
    for (int row = 0; row < 4; ++row) {
        std::array<std::uint8_t, 4> values{};
        for (int col = 0; col < 4; ++col) values[col] = board.cells[row * 4 + col];
        score += g_heuristics[make_line(values)];
    }
    for (int col = 0; col < 4; ++col) {
        std::array<std::uint8_t, 4> values{};
        for (int row = 0; row < 4; ++row) values[row] = board.cells[row * 4 + col];
        score += g_heuristics[make_line(values)];
    }
    return score;
}

using TupleIndices = std::array<std::array<std::uint32_t, kTupleCount>, kSymmetryCount>;

TupleIndices make_tuple_indices(const Board& board) noexcept {
    TupleIndices indices{};
    for (int symmetry = 0; symmetry < kSymmetryCount; ++symmetry) {
        for (int pattern = 0; pattern < kTupleCount; ++pattern) {
            std::uint32_t index = 0;
            for (int cell = 0; cell < kTupleSize; ++cell) {
                const std::uint32_t rank = std::min<std::uint32_t>(
                    board.cells[kPatternCellMaps[symmetry][pattern][cell]], 15U
                );
                index = (index << 4U) | rank;
            }
            indices[symmetry][pattern] = index;
        }
    }
    return indices;
}

float ntuple_afterstate_value(const Board& board, const float* tables) {
    const TupleIndices indices = make_tuple_indices(board);
    double total = 0.0;
    for (int symmetry = 0; symmetry < kSymmetryCount; ++symmetry) {
        for (int pattern = 0; pattern < kTupleCount; ++pattern) {
            total += tables[static_cast<std::size_t>(pattern) * kTupleTableSize +
                            indices[symmetry][pattern]];
        }
    }
    return static_cast<float>(total / 8.0);
}

float ntuple_state_value(const LegalMoves& moves, const float* tables) {
    if (moves.empty()) return 0.0f;
    float best = -std::numeric_limits<float>::infinity();
    for (const auto& move : moves) {
        best = std::max(best, static_cast<float>(move.result.reward) +
                               ntuple_afterstate_value(move.result.board, tables));
    }
    return best;
}

float ntuple_state_value(const Board& board, const float* tables) {
    const auto moves = legal_moves(board);
    return ntuple_state_value(moves, tables);
}

double ntuple_afterstate_update(const Board& board, float* tables,
                                double target, double alpha) {
    const TupleIndices indices = make_tuple_indices(board);
    double total = 0.0;
    for (int symmetry = 0; symmetry < kSymmetryCount; ++symmetry) {
        for (int pattern = 0; pattern < kTupleCount; ++pattern) {
            total += tables[static_cast<std::size_t>(pattern) * kTupleTableSize +
                            indices[symmetry][pattern]];
        }
    }

    const double error = target - total / static_cast<double>(kSymmetryCount);
    // ntuple_afterstate_value averages the 8 symmetry views; keep the TD
    // update consistent with that value-function normalization.
    const float increment = static_cast<float>(alpha * error / kSymmetryCount);
    for (int symmetry = 0; symmetry < kSymmetryCount; ++symmetry) {
        for (int pattern = 0; pattern < kTupleCount; ++pattern) {
            tables[static_cast<std::size_t>(pattern) * kTupleTableSize + indices[symmetry][pattern]] += increment;
        }
    }
    return error;
}

double ntuple_td_lambda_update(
    const Board& board, float* tables, double target, double alpha,
    double gamma, double lambda,
    std::unordered_map<std::uint32_t, double>& traces) {
    if (lambda <= 0.0) {
        return ntuple_afterstate_update(board, tables, target, alpha);
    }

    const double error = target - ntuple_afterstate_value(board, tables);
    const double decay = gamma * lambda;
    constexpr double kTracePrune = 1.0e-7;
    for (auto it = traces.begin(); it != traces.end();) {
        it->second *= decay;
        if (std::abs(it->second) < kTracePrune) {
            it = traces.erase(it);
        } else {
            ++it;
        }
    }

    const TupleIndices indices = make_tuple_indices(board);
    constexpr double kFeatureScale = 1.0 / kSymmetryCount;
    for (int symmetry = 0; symmetry < kSymmetryCount; ++symmetry) {
        for (int pattern = 0; pattern < kTupleCount; ++pattern) {
            const std::uint32_t feature = static_cast<std::uint32_t>(
                static_cast<std::size_t>(pattern) * kTupleTableSize +
                indices[symmetry][pattern]);
            traces[feature] += kFeatureScale;
        }
    }

    const float scaled_error = static_cast<float>(alpha * error);
    for (const auto& trace : traces) {
        const std::uint32_t pattern = trace.first / kTupleTableSize;
        const std::uint32_t index = trace.first % kTupleTableSize;
        tables[static_cast<std::size_t>(pattern) * kTupleTableSize + index] +=
            scaled_error * static_cast<float>(trace.second);
    }
    return error;
}

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

double random_unit(std::mt19937_64& rng) {
    return std::generate_canonical<double, 53>(rng);
}

bool spawn_random_tile(Board& board, std::mt19937_64& rng) {
    std::array<int, 16> empty{};
    int count = 0;
    for (int cell = 0; cell < 16; ++cell) {
        if (board.cells[cell] == 0) empty[count++] = cell;
    }
    if (count == 0) return false;
    std::uniform_int_distribution<int> position_distribution(0, count - 1);
    const int cell = empty[position_distribution(rng)];
    board.cells[cell] = random_unit(rng) < 0.9 ? 1 : 2;
    return true;
}

struct TrainingStats {
    std::uint64_t score = 0;
    std::uint64_t moves = 0;
    std::uint64_t merges = 0;
    std::uint8_t max_rank = 0;
    double mean_abs_td_error = 0.0;
};

std::pair<int, MoveResult> best_ntuple_afterstate(
    const LegalMoves& moves, const float* tables) {
    int best_action = -1;
    double best_value = -std::numeric_limits<double>::infinity();
    MoveResult best_result{};
    for (const auto& move : moves) {
        const double value = static_cast<double>(move.result.reward) +
                             ntuple_afterstate_value(move.result.board, tables);
        if (value > best_value + 1e-9) {
            best_value = value;
            best_action = move.action;
            best_result = move.result;
        }
    }
    if (best_action < 0) return {-1, best_result};
    return {best_action, best_result};
}

TrainingStats train_ntuple_episode(
    float* tables, std::uint64_t seed, double epsilon, double alpha,
    double gamma, double td_lambda, std::uint64_t max_moves, const std::uint8_t* initial_ranks,
    bool has_initial_board, int capture_min_rank, int capture_every,
    std::uint8_t* captured_positions, std::uint32_t capture_capacity,
    std::uint32_t* captured_count) {
    std::mt19937_64 spawn_rng(splitmix64(seed));
    std::mt19937_64 action_rng(splitmix64(seed ^ 0xD1B54A32D192ED03ULL));
    Board board{};
    if (has_initial_board) {
        board = board_from_ranks(initial_ranks);
    } else {
        spawn_random_tile(board, spawn_rng);
        spawn_random_tile(board, spawn_rng);
    }

    TrainingStats stats;
    double td_error_sum = 0.0;
    std::uint64_t td_updates = 0;
    Board previous_afterstate{};
    bool has_previous_afterstate = false;
    std::unordered_map<std::uint32_t, double> eligibility_traces;

    for (std::uint64_t move = 0; move < max_moves; ++move) {
        const auto moves = legal_moves(board);
        if (moves.empty()) {
            if (has_previous_afterstate) {
                const double td_error = ntuple_td_lambda_update(
                    previous_afterstate, tables, 0.0, alpha, gamma, td_lambda,
                    eligibility_traces
                );
                td_error_sum += std::abs(td_error);
                ++td_updates;
            }
            break;
        }

        MoveResult afterstate;
        if (random_unit(action_rng) < epsilon) {
            std::uniform_int_distribution<std::size_t> action_distribution(0, moves.size() - 1);
            const auto& selected = moves.entries[action_distribution(action_rng)];
            afterstate = selected.result;
        } else {
            auto selected = best_ntuple_afterstate(moves, tables);
            afterstate = selected.second;
        }

        // On-policy afterstate TD: choose the action that will actually be
        // executed, then update the previous afterstate from this transition.
        // Do not preselect a hypothetical next action and later reselect it
        // after the weight update; that creates a target/policy mismatch.
        if (has_previous_afterstate) {
            const double target = static_cast<double>(afterstate.reward) +
                                  gamma * ntuple_afterstate_value(afterstate.board, tables);
            const double td_error = ntuple_td_lambda_update(
                previous_afterstate, tables, target, alpha, gamma, td_lambda,
                eligibility_traces
            );
            td_error_sum += std::abs(td_error);
            ++td_updates;
        }
        previous_afterstate = afterstate.board;
        has_previous_afterstate = true;

        stats.score += afterstate.reward;
        stats.merges += static_cast<std::uint64_t>(afterstate.merges);
        ++stats.moves;
        board = afterstate.board;
        spawn_random_tile(board, spawn_rng);

        if (captured_positions && captured_count && capture_every > 0 &&
            stats.moves % static_cast<std::uint64_t>(capture_every) == 0 &&
            *captured_count < capture_capacity) {
            std::uint8_t maximum_rank = 0;
            for (std::uint8_t rank : board.cells) maximum_rank = std::max(maximum_rank, rank);
            if (maximum_rank >= capture_min_rank && !legal_moves(board).empty()) {
                std::uint8_t* destination = captured_positions +
                    static_cast<std::size_t>(*captured_count) * 16;
                std::copy(board.cells.begin(), board.cells.end(), destination);
                ++(*captured_count);
            }
        }
    }

    for (std::uint8_t rank : board.cells) stats.max_rank = std::max(stats.max_rank, rank);
    stats.mean_abs_td_error = td_error_sum / static_cast<double>(std::max<std::uint64_t>(1, td_updates));
    return stats;
}

struct CacheKey {
    Board board;
    int depth = 0;
    bool chance = false;
    bool operator==(const CacheKey& other) const noexcept {
        return depth == other.depth && chance == other.chance && board == other.board;
    }
};

struct CacheHash {
    std::size_t operator()(const CacheKey& key) const noexcept {
        std::uint64_t hash = 1469598103934665603ULL;
        for (std::uint8_t value : key.board.cells) {
            hash ^= value;
            hash *= 1099511628211ULL;
        }
        hash ^= static_cast<std::uint64_t>(key.depth + (key.chance ? 0x10000 : 0));
        hash *= 1099511628211ULL;
        return static_cast<std::size_t>(hash);
    }
};

struct CacheEntry {
    CacheKey key{};
    double value = 0.0;
    bool occupied = false;
};

struct CacheBucket {
    std::array<CacheEntry, kCacheWays> entries{};
    std::uint8_t next_victim = 0;
};

// Fixed-capacity 4-way set-associative cache. Unlike unordered_map, lookups
// and replacements allocate nothing and cannot trigger a long rehash/clear.
class TranspositionCache {
public:
    TranspositionCache() : buckets_(kCacheBucketCount) {}

    bool find(const CacheKey& key, double& value) const noexcept {
        const CacheBucket& bucket = buckets_[hasher_(key) & (kCacheBucketCount - 1)];
        for (const CacheEntry& entry : bucket.entries) {
            if (entry.occupied && entry.key == key) {
                value = entry.value;
                return true;
            }
        }
        return false;
    }

    void insert(const CacheKey& key, double value) noexcept {
        CacheBucket& bucket = buckets_[hasher_(key) & (kCacheBucketCount - 1)];
        for (CacheEntry& entry : bucket.entries) {
            if (entry.occupied && entry.key == key) {
                entry.value = value;
                return;
            }
            if (!entry.occupied) {
                entry.key = key;
                entry.value = value;
                entry.occupied = true;
                return;
            }
        }

        CacheEntry& victim = bucket.entries[bucket.next_victim];
        victim.key = key;
        victim.value = value;
        victim.occupied = true;
        bucket.next_victim = static_cast<std::uint8_t>((bucket.next_victim + 1) % kCacheWays);
    }

    void clear() noexcept {
        for (CacheBucket& bucket : buckets_) {
            for (CacheEntry& entry : bucket.entries) entry.occupied = false;
            bucket.next_victim = 0;
        }
    }

private:
    std::vector<CacheBucket> buckets_;
    CacheHash hasher_;
};

struct SearchContext {
    TranspositionCache player_cache;
    TranspositionCache chance_cache;

    void store(const CacheKey& key, double value) noexcept {
        if (key.chance) chance_cache.insert(key, value);
        else player_cache.insert(key, value);
    }

    void clear() noexcept {
        player_cache.clear();
        chance_cache.clear();
    }
};

struct SearchControl {
    SearchContext& context;
    const float* tables = nullptr;
    bool use_ntuple = false;
    std::uint64_t node_budget = 0;
    std::uint64_t nodes = 0;
    std::uint64_t cache_hits = 0;
    std::chrono::steady_clock::time_point deadline;
    bool has_deadline = false;

    explicit SearchControl(SearchContext& search_context) : context(search_context) {}

    void visit() {
        ++nodes;
        if (nodes > node_budget ||
            (has_deadline && (nodes & 31U) == 0 &&
             std::chrono::steady_clock::now() >= deadline)) {
            throw 1;
        }
    }

    double leaf_value(const Board& board, const LegalMoves& moves) const {
        return use_ntuple ? ntuple_state_value(moves, tables) : classic_heuristic(board);
    }

    double player_value(const Board& board, int depth) {
        const CacheKey key{board, depth, false};
        double cached_value = 0.0;
        if (context.player_cache.find(key, cached_value)) {
            ++cache_hits;
            return cached_value;
        }
        visit();
        const auto moves = legal_moves(board);
        if (moves.empty()) {
            context.store(key, 0.0);
            return 0.0;
        }
        if (depth <= 0) {
            const double value = leaf_value(board, moves);
            context.store(key, value);
            return value;
        }

        double best = -std::numeric_limits<double>::infinity();
        for (const auto& move : moves) {
            const double continuation = chance_value(move.result.board, depth - 1);
            const double value = continuation +
                (use_ntuple ? static_cast<double>(move.result.reward) : 0.0);
            best = std::max(best, value);
        }
        context.store(key, best);
        return best;
    }

    double chance_value(const Board& afterstate, int depth) {
        const CacheKey key{afterstate, depth, true};
        double cached_value = 0.0;
        if (context.chance_cache.find(key, cached_value)) {
            ++cache_hits;
            return cached_value;
        }
        visit();
        const int empty_count = count_empty(afterstate);
        if (empty_count == 0) {
            const double value = player_value(afterstate, depth);
            context.store(key, value);
            return value;
        }

        const double cell_probability = 1.0 / static_cast<double>(empty_count);
        double expected = 0.0;
        for (int cell = 0; cell < 16; ++cell) {
            if (afterstate.cells[cell] != 0) continue;
            Board with_two = afterstate;
            with_two.cells[cell] = 1;
            expected += cell_probability * 0.9 * player_value(with_two, depth);

            Board with_four = afterstate;
            with_four.cells[cell] = 2;
            expected += cell_probability * 0.1 * player_value(with_four, depth);
        }
        context.store(key, expected);
        return expected;
    }
};

Board board_from_ranks(const std::uint8_t* ranks) {
    Board board;
    std::copy(ranks, ranks + 16, board.cells.begin());
    for (auto& rank : board.cells) rank = std::min<std::uint8_t>(rank, 31);
    return board;
}

Board downgrade_root(Board board, int threshold_rank) {
    const int maximum = *std::max_element(board.cells.begin(), board.cells.end());
    if (maximum < threshold_rank || maximum <= 1) return board;
    int missing_rank = -1;
    for (int rank = maximum - 1; rank >= 1; --rank) {
        if (std::find(board.cells.begin(), board.cells.end(), static_cast<std::uint8_t>(rank)) == board.cells.end()) {
            missing_rank = rank;
            break;
        }
    }
    if (missing_rank < 0) return board;
    for (auto& rank : board.cells) {
        if (rank > missing_rank) --rank;
    }
    return board;
}

}  // namespace

extern "C" int destroyer_apply_move(const std::uint8_t* input, int action,
                                    std::uint8_t* output, std::uint64_t* reward, int* merges) {
    initialize_tables();
    const Board board = board_from_ranks(input);
    const auto result = apply_move(board, action);
    std::copy(result.board.cells.begin(), result.board.cells.end(), output);
    if (reward) *reward = result.reward;
    if (merges) *merges = result.merges;
    return result.moved ? 1 : 0;
}

extern "C" float destroyer_classic_heuristic(const std::uint8_t* input) {
    initialize_tables();
    return classic_heuristic(board_from_ranks(input));
}

extern "C" float destroyer_ntuple_afterstate_value(const std::uint8_t* input,
                                                    const float* tables) {
    if (!input || !tables) return 0.0f;
    return ntuple_afterstate_value(board_from_ranks(input), tables);
}

extern "C" float destroyer_ntuple_state_value(const std::uint8_t* input,
                                               const float* tables) {
    if (!input || !tables) return 0.0f;
    initialize_tables();
    return ntuple_state_value(board_from_ranks(input), tables);
}

extern "C" void* destroyer_search_create() {
    initialize_tables();
    return new SearchContext();
}

extern "C" void destroyer_search_destroy(void* raw_context) {
    delete static_cast<SearchContext*>(raw_context);
}

extern "C" void destroyer_search_clear(void* raw_context) {
    auto* context = static_cast<SearchContext*>(raw_context);
    if (!context) return;
    context->clear();
}

extern "C" int destroyer_search_choose(
    void* raw_context,
    const std::uint8_t* input,
    const float* ntuple_tables,
    int use_ntuple,
    int max_depth,
    std::uint64_t node_budget,
    double time_budget_ms,
    int downgrade_threshold_rank,
    int downgrade_steps,
    int* out_action,
    double* out_value,
    int* out_depth,
    std::uint64_t* out_nodes,
    std::uint64_t* out_cache_hits,
    int* out_complete
) {
    if (!raw_context || !input || !out_action || !out_value || !out_depth ||
        !out_nodes || !out_cache_hits || !out_complete || max_depth < 1 ||
        downgrade_threshold_rank < 0 || downgrade_threshold_rank > 31 || downgrade_steps < 0 ||
        (downgrade_threshold_rank > 0 && downgrade_steps != 1)) return 0;
    auto& context = *static_cast<SearchContext*>(raw_context);
    const Board original_board = board_from_ranks(input);
    SearchControl control{context};
    control.tables = ntuple_tables;
    control.use_ntuple = use_ntuple != 0 && ntuple_tables != nullptr;
    control.node_budget = node_budget ? node_budget : 1;
    const Board board = control.use_ntuple && downgrade_threshold_rank > 0
        ? downgrade_root(original_board, downgrade_threshold_rank)
        : original_board;
    const auto moves = legal_moves(board);
    if (moves.empty()) return 0;
    control.has_deadline = time_budget_ms > 0.0;
    if (control.has_deadline) {
        control.deadline = std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double, std::milli>(time_budget_ms));
    }

    int best_action = moves.front().action;
    double best_value = -std::numeric_limits<double>::infinity();
    for (const auto& move : moves) {
        const auto& afterstate = move.result;
        double value = 0.0;
        if (control.use_ntuple) {
            value = static_cast<double>(afterstate.reward) + ntuple_afterstate_value(afterstate.board, ntuple_tables);
        } else {
            value = static_cast<double>(classic_heuristic(afterstate.board));
        }
        if (value > best_value) {
            best_value = value;
            best_action = move.action;
        }
    }

    int completed_depth = 0;
    bool complete = false;
    for (int depth = 1; depth <= max_depth; ++depth) {
        int depth_best_action = moves.front().action;
        double depth_best_value = -std::numeric_limits<double>::infinity();
        try {
            for (const auto& move : moves) {
                const auto& afterstate = move.result;
                const double continuation = control.chance_value(afterstate.board, depth - 1);
                const double value = continuation +
                    (control.use_ntuple ? static_cast<double>(afterstate.reward) : 0.0);
                if (value > depth_best_value) {
                    depth_best_value = value;
                    depth_best_action = move.action;
                }
            }
        } catch (int) {
            break;
        }
        completed_depth = depth;
        best_action = depth_best_action;
        best_value = depth_best_value;
        if (depth == max_depth) complete = true;
    }

    *out_action = best_action;
    *out_value = best_value;
    *out_depth = completed_depth;
    *out_nodes = control.nodes;
    *out_cache_hits = control.cache_hits;
    *out_complete = complete ? 1 : 0;
    return 1;
}

extern "C" int destroyer_ntuple_train_episode(
    float* tables,
    std::uint64_t seed,
    double epsilon,
    double alpha,
    double gamma,
    std::uint64_t max_moves,
    const std::uint8_t* initial_ranks,
    int has_initial_board,
    int capture_min_rank,
    int capture_every,
    std::uint8_t* captured_positions,
    std::uint32_t capture_capacity,
    std::uint32_t* captured_count,
    std::uint64_t* out_score,
    std::uint64_t* out_moves,
    std::uint64_t* out_merges,
    std::uint8_t* out_max_rank,
    double* out_mean_abs_td_error
) {
    if (!tables || !out_score || !out_moves || !out_merges || !out_max_rank ||
        !out_mean_abs_td_error || epsilon < 0.0 || epsilon > 1.0 ||
        alpha <= 0.0 || alpha > 1.0 || gamma < 0.0 || gamma > 1.0 ||
        max_moves == 0 || (has_initial_board && !initial_ranks) ||
        capture_min_rank < 1 || capture_min_rank > 31 || capture_every < 1 ||
        (captured_positions && !captured_count)) return 0;

    if (captured_count) *captured_count = 0;

    initialize_tables();
    const TrainingStats stats = train_ntuple_episode(
        tables, seed, epsilon, alpha, gamma, 0.0, max_moves, initial_ranks,
        has_initial_board != 0, capture_min_rank, capture_every,
        captured_positions, capture_capacity, captured_count
    );
    *out_score = stats.score;
    *out_moves = stats.moves;
    *out_merges = stats.merges;
    *out_max_rank = stats.max_rank;
    *out_mean_abs_td_error = stats.mean_abs_td_error;
    return 1;
}

extern "C" int destroyer_ntuple_train_episode_lambda(
    float* tables, std::uint64_t seed, double epsilon, double alpha,
    double gamma, double td_lambda, std::uint64_t max_moves,
    const std::uint8_t* initial_ranks, int has_initial_board,
    int capture_min_rank, int capture_every, std::uint8_t* captured_positions,
    std::uint32_t capture_capacity, std::uint32_t* captured_count,
    std::uint64_t* out_score, std::uint64_t* out_moves,
    std::uint64_t* out_merges, std::uint8_t* out_max_rank,
    double* out_mean_abs_td_error) {
    if (!tables || !out_score || !out_moves || !out_merges || !out_max_rank ||
        !out_mean_abs_td_error || epsilon < 0.0 || epsilon > 1.0 ||
        alpha <= 0.0 || alpha > 1.0 || gamma < 0.0 || gamma > 1.0 ||
        td_lambda < 0.0 || td_lambda > 1.0 || max_moves == 0 ||
        (has_initial_board && !initial_ranks) || capture_min_rank < 1 ||
        capture_min_rank > 31 || capture_every < 1 ||
        (captured_positions && !captured_count)) return 0;

    if (captured_count) *captured_count = 0;
    initialize_tables();
    const TrainingStats stats = train_ntuple_episode(
        tables, seed, epsilon, alpha, gamma, td_lambda, max_moves,
        initial_ranks, has_initial_board != 0, capture_min_rank,
        capture_every, captured_positions, capture_capacity, captured_count);
    *out_score = stats.score;
    *out_moves = stats.moves;
    *out_merges = stats.merges;
    *out_max_rank = stats.max_rank;
    *out_mean_abs_td_error = stats.mean_abs_td_error;
    return 1;
}
