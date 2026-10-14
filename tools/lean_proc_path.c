/* Linux sandbox compatibility only: Lean 4.19 asks for /proc/<pid>/exe,
   while this host exposes /proc/self/exe. No Lean/kernel code is changed. */
#define _GNU_SOURCE
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <dlfcn.h>
ssize_t readlink(const char *path, char *buf, size_t n) {
    ssize_t (*next)(const char *, char *, size_t) = dlsym(RTLD_NEXT, "readlink");
    ssize_t result = next(path, buf, n);
    if (result == -1 && strncmp(path, "/proc/", 6) == 0 &&
        strstr(path, "/exe") != NULL && strcmp(path, "/proc/self/exe") != 0) {
        return next("/proc/self/exe", buf, n);
    }
    return result;
}
