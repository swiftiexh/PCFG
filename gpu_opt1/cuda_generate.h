#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void cuda_generate_guesses(
    const char* prefix,
    char** h_values,
    int n,
    char** h_results);

#ifdef __cplusplus
}
#endif