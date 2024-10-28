#ifndef NS3_PACKERL_TYPES_H
#define NS3_PACKERL_TYPES_H

#include <functional>
#include <tuple>
#include <utility>


using std::pair, std::hash, std::tuple;

typedef pair<uint32_t, uint32_t> U32Pair;

/**
 * Hash function for pairs.
 * @tparam T1 first element
 * @tparam T2 second element
 */
struct pair_hash
{
    template <class T1, class T2>
    size_t operator() (const pair<T1, T2> &pair) const
    {
        auto hash1 = hash<T1>()(pair.first);
        auto hash2 = hash<T2>()(pair.second);
        return hash1 == hash2 ? hash1 : hash1 ^ hash2;
    }
};

typedef tuple<uint32_t, uint32_t, uint32_t> U32Triple;

/**
 * Hash function for triples.
 */
struct triple_hash
{
    template <class T1, class T2, class T3>
    size_t operator() (const std::tuple<T1, T2, T3> &triple) const
    {
        auto hash1 = std::hash<T1>()(std::get<0>(triple));
        auto hash2 = std::hash<T2>()(std::get<1>(triple));
        auto hash3 = std::hash<T3>()(std::get<2>(triple));

        // use boost's hash_combine with reciprocal of golden ratio as magic number (TODO is there a better way?)
        size_t seed = hash1;
        seed ^= hash2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hash3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);

        return seed;
    }
};


#endif // NS3_PACKERL_TYPES_H
