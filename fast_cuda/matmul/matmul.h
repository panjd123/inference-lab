#include <string>

enum class FAST_MATMUL {
    NATIVE,
    SHARED_MEMORY,
    SHARED_MEMORY_NATIVE,
    TILE,
};

inline std::string fast_matmul_method_to_string(FAST_MATMUL method) {
    switch (method) {
        case FAST_MATMUL::NATIVE:
            return "NATIVE";
        case FAST_MATMUL::SHARED_MEMORY:
            return "SHARED_MEMORY";
        case FAST_MATMUL::SHARED_MEMORY_NATIVE:
            return "SHARED_MEMORY_NATIVE";
        default:
            return "UNKNOWN";
    }
}