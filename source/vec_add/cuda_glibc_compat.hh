#pragma once

#if defined(__CUDACC__) && defined(__GLIBC__) && !defined(_GNU_SOURCE)

#include <pthread.h>
#include <time.h>

extern "C" {
int pthread_cond_clockwait(pthread_cond_t* condition, pthread_mutex_t* mutex, clockid_t clock,
                           const timespec* timeout) noexcept;
int pthread_mutex_clocklock(pthread_mutex_t* mutex, clockid_t clock,
                            const timespec* timeout) noexcept;
}

#endif
