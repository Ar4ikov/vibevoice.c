/**
 * @file vv_spawn.h
 * @brief Run a helper program with an argument vector, never a shell.
 *
 * Internal header. The runtime starts two kinds of helper: ffmpeg, to decode
 * audio that is not WAV and to list capture devices, and a recorder (ffmpeg
 * or arecord) for the microphone. Both used to go through popen(), which
 * hands a single string to `/bin/sh -c` or `cmd.exe /c` -- so a path or a
 * device name containing a quote, `$(...)`, a backtick, `;`, `&` or `%VAR%`
 * was executed rather than passed (CodeQL cpp/command-line-injection).
 *
 * Here the arguments reach the child exactly as given. On POSIX the child is
 * started with posix_spawnp(); on Windows with CreateProcessW(), after
 * quoting each argument by the rules the C runtime uses to split the command
 * line back into argv, and with only the pipe and the null device inherited.
 *
 * The child's stdout comes back as a FILE* for reading. Its stdin is the
 * null device; its stderr goes wherever the caller says: the null device,
 * the same pipe as stdout (ffmpeg prints device lists on stderr), or this
 * process's own stderr (a recorder's complaints about a missing device are
 * worth seeing).
 */
#ifndef VV_SPAWN_H
#define VV_SPAWN_H

#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vv_child vv_child_t;

/** @brief Where the child's stderr goes. */
typedef enum {
    VV_SPAWN_STDERR_NULL = 0,  /**< discarded                       */
    VV_SPAWN_STDERR_MERGE,     /**< into the stdout pipe             */
    VV_SPAWN_STDERR_INHERIT,   /**< this process's stderr            */
} vv_spawn_stderr_t;

/**
 * @brief Start `argv[0]`, looked up on PATH, with arguments `argv`.
 *
 * @param argv          NULL-terminated argument vector; argv[0] is the
 *                      program name. Arguments are passed through verbatim.
 * @param err           Where the child's stderr goes.
 * @param out           Receives the read end of the child's stdout, in
 *                      binary mode. Owned by the child handle: close it only
 *                      through vv_spawn_wait().
 * @return The child, or NULL if it could not be started (program not found,
 *         out of resources).
 */
vv_child_t* vv_spawn_read(const char* const argv[], vv_spawn_stderr_t err,
                          FILE** out);

/**
 * @brief Ask the child to stop (SIGTERM, or TerminateProcess on Windows)
 * without touching its stdout.
 *
 * For a reader blocked in fread() on another thread: the child exits, the
 * read returns end-of-file, and the thread can be joined before
 * vv_spawn_wait() closes the stream it was reading.
 */
void vv_spawn_terminate(vv_child_t* child);

/**
 * @brief Close the child's stdout, wait for it to exit, and free the handle.
 *
 * @param terminate  Ask the child to stop first (SIGTERM, or
 *                   TerminateProcess on Windows). For a recorder that would
 *                   otherwise run until it next tries to write.
 * @return The child's exit status, 128+signal if it was killed by a signal,
 *         or -1 if the status could not be obtained.
 */
int vv_spawn_wait(vv_child_t* child, bool terminate);

#ifdef __cplusplus
}
#endif

#endif /* VV_SPAWN_H */
