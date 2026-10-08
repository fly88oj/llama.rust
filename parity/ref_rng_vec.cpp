// Reference vectors for the perplexity scorers' task-selection randomness
// (std::mt19937 + libstdc++ 13's uniform_int_distribution Lemire branch + the
// raw `int(scale*rng()*aux.size())` idiom), printed by the same g++ 13
// toolchain that built /home/jeffrey/llm/llama.cpp/build-rust-ref.
//
//   g++ -O2 -std=c++17 -o ref_rng_vec ref_rng_vec.cpp && ./ref_rng_vec
//
// The outputs are pinned in crates/tools/perplexity/src/scorers.rs tests
// (mt19937_reference_vectors / lemire_matches_libstdcpp_vectors /
// scale_draw_matches_libstdcpp_vectors).
#include <random>
#include <vector>
#include <cstdio>

int main() {
    {
        std::mt19937 rng(1);
        std::uniform_int_distribution<size_t> dist(0, 9);
        printf("lemire:");
        for (int i = 0; i < 10; i++) printf(" %zu", dist(rng));
        printf("\n");
    }
    {
        std::mt19937 rng(1);
        printf("raw:");
        for (int i = 0; i < 5; i++) printf(" %lu", (unsigned long) rng());
        printf("\n");
    }
    {
        std::mt19937 rng(1);
        std::vector<int> aux(5);
        for (int i = 0; i < 5; i++) aux[i] = i;
        float scale = 1 / (1.f + (float) rng.max());
        printf("scale:");
        for (int i = 0; i < 5; i++) printf(" %d", int(scale * rng() * aux.size()));
        printf("\n");
    }
    {
        std::mt19937 rng(1);
        std::uniform_int_distribution<size_t> dist(0, 59);
        printf("hswag60:");
        for (int i = 0; i < 8; i++) printf(" %zu", dist(rng));
        printf("\n");
    }
    {
        std::mt19937 rng(5489);
        printf("seed5489:");
        for (int i = 0; i < 5; i++) printf(" %lu", (unsigned long) rng());
        printf("\n");
    }
    {
        std::mt19937 rng(1);
        for (int i = 0; i < 624; i++) rng();
        printf("after624: %lu\n", (unsigned long) rng());
    }
    return 0;
}
