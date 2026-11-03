#pragma once
// Internal helpers for the model file readers/writers. Include from .cpp files only.
#include <cstdint>
#include <fstream>

namespace dod_detail {

template <typename T>
inline void put(std::ofstream& f, const T& v) {
	f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
inline void get(std::ifstream& f, T& v) {
	f.read(reinterpret_cast<char*>(&v), sizeof(T));
}

// out = a * b, but only if the product is <= limit (no overflow possible).
inline bool mulWithin(std::uint64_t a, std::uint64_t b, std::uint64_t limit, std::uint64_t& out) {
	if (a == 0 || b == 0) { out = 0; return true; }
	if (a > limit / b) return false;
	out = a * b;
	return true;
}

} // namespace dod_detail
