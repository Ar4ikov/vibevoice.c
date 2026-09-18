/**
 * @file spawn_child.c
 * @brief Helper for test_untrusted_input: echoes its arguments.
 *
 * Writes every argument after argv[0] to stdout followed by a NUL byte, so
 * the test can check that each one arrived exactly as it was passed. A
 * leading control argument changes the behaviour:
 *
 *   --exit7    exit with status 7
 *   --stderr   also write "on-stderr" to stderr
 *   --sleep    sleep for a minute (to be terminated)
 */
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    int first = 1;
    if (argc > 1 && strcmp(argv[1], "--exit7") == 0) return 7;
    if (argc > 1 && strcmp(argv[1], "--stderr") == 0) {
        fputs("on-stderr", stderr);
        fflush(stderr);
        first = 2;
    }
    if (argc > 1 && strcmp(argv[1], "--sleep") == 0) {
        fputs("sleeping", stdout);
        fputc('\0', stdout);
        fflush(stdout);
#ifdef _WIN32
        Sleep(60000);
#else
        sleep(60);
#endif
        return 0;
    }
    for (int i = first; i < argc; i++) {
        fputs(argv[i], stdout);
        fputc('\0', stdout);
    }
    return 0;
}
