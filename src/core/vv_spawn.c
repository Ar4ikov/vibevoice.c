/**
 * @file vv_spawn.c
 * @brief Start helper programs from an argument vector -- see vv_spawn.h.
 */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE  /* pipe2 */
#endif

#include "vv_spawn.h"
#include "vibevoice/vibevoice.h"

#include <string.h>

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <fcntl.h>
#include <io.h>

struct vv_child {
    HANDLE process;
    FILE*  out;
};

/* Growable UTF-16 buffer for the command line. */
typedef struct {
    wchar_t* p;
    size_t   len, cap;
    bool     failed;
} wbuf_t;

static void wb_put(wbuf_t* b, const wchar_t* s, size_t n) {
    if (b->failed) return;
    if (b->len + n + 1 > b->cap) {
        size_t ncap = b->cap ? b->cap * 2 : 256;
        while (ncap < b->len + n + 1) ncap *= 2;
        wchar_t* np = (wchar_t*)vv_alloc(ncap * sizeof(wchar_t));
        if (!np) { b->failed = true; return; }
        if (b->p) {
            memcpy(np, b->p, b->len * sizeof(wchar_t));
            vv_free(b->p);
        }
        b->p = np;
        b->cap = ncap;
    }
    memcpy(b->p + b->len, s, n * sizeof(wchar_t));
    b->len += n;
    b->p[b->len] = L'\0';
}

static void wb_repeat(wbuf_t* b, wchar_t c, size_t n) {
    for (size_t i = 0; i < n; i++) wb_put(b, &c, 1);
}

/*
 * Append one argument so that the C runtime's command-line parser (the one
 * CommandLineToArgvW implements) hands it back unchanged: quoted when it is
 * empty or holds whitespace or a quote; inside quotes a run of backslashes is
 * doubled when a quote or the closing quote follows it, and a quote is
 * escaped. No cmd.exe is involved, so `&`, `|`, `^`, `%` need nothing.
 */
static void wb_arg(wbuf_t* b, const wchar_t* a) {
    if (a[0] && !wcspbrk(a, L" \t\n\v\"")) {
        wb_put(b, a, wcslen(a));
        return;
    }
    wb_put(b, L"\"", 1);
    for (const wchar_t* p = a;; p++) {
        size_t slashes = 0;
        while (*p == L'\\') { slashes++; p++; }
        if (*p == L'\0') {
            wb_repeat(b, L'\\', slashes * 2);
            break;
        }
        if (*p == L'"') {
            wb_repeat(b, L'\\', slashes * 2 + 1);
            wb_put(b, L"\"", 1);
        } else {
            wb_repeat(b, L'\\', slashes);
            wb_put(b, p, 1);
        }
    }
    wb_put(b, L"\"", 1);
}

static HANDLE merge_or_null(vv_spawn_stderr_t err, HANDLE wr, HANDLE nul) {
    return err == VV_SPAWN_STDERR_MERGE ? wr : nul;
}

/*
 * Arguments arrive in the process code page: argv from main(), getenv() and
 * GetTempPathA() all speak it, and the child gets the same characters back
 * whichever way it reads its command line.
 */
static wchar_t* widen(const char* s) {
    const int n = MultiByteToWideChar(CP_ACP, 0, s, -1, NULL, 0);
    if (n <= 0) return NULL;
    wchar_t* w = (wchar_t*)vv_alloc((size_t)n * sizeof(wchar_t));
    if (!w) return NULL;
    if (MultiByteToWideChar(CP_ACP, 0, s, -1, w, n) != n) {
        vv_free(w);
        return NULL;
    }
    return w;
}

vv_child_t* vv_spawn_read(const char* const argv[], vv_spawn_stderr_t err,
                          FILE** out) {
    if (!argv || !argv[0] || !argv[0][0] || !out) return NULL;
    *out = NULL;
    /* argv[0] is parsed by different rules; a quote in it cannot round-trip. */
    if (strchr(argv[0], '"')) return NULL;

    wbuf_t cmd = {0};
    for (int i = 0; argv[i]; i++) {
        wchar_t* w = widen(argv[i]);
        if (!w) { cmd.failed = true; break; }
        if (i) wb_put(&cmd, L" ", 1);
        wb_arg(&cmd, w);
        vv_free(w);
    }
    if (cmd.failed || !cmd.p || cmd.len >= 32767) {
        vv_free(cmd.p);
        return NULL;
    }

    SECURITY_ATTRIBUTES sa = { sizeof(sa), NULL, TRUE };
    HANDLE rd = NULL, wr = NULL;
    if (!CreatePipe(&rd, &wr, &sa, 0)) { vv_free(cmd.p); return NULL; }
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ | GENERIC_WRITE,
                             FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                             OPEN_EXISTING, 0, NULL);
    if (nul == INVALID_HANDLE_VALUE) {
        CloseHandle(rd); CloseHandle(wr); vv_free(cmd.p);
        return NULL;
    }

    /*
     * The caller's own stderr, when asked for: an inheritable duplicate, since
     * the original may not be. A process with no console has none, and the
     * child then writes to NUL instead.
     */
    HANDLE errh = merge_or_null(err, wr, nul);
    HANDLE errdup = NULL;
    if (err == VV_SPAWN_STDERR_INHERIT) {
        HANDLE mine = GetStdHandle(STD_ERROR_HANDLE);
        if (mine && mine != INVALID_HANDLE_VALUE &&
            DuplicateHandle(GetCurrentProcess(), mine, GetCurrentProcess(),
                            &errdup, 0, TRUE, DUPLICATE_SAME_ACCESS))
            errh = errdup;
    }

    /*
     * Only these handles are inherited. Without the list the child would get
     * every inheritable handle in the process -- including the write end of
     * another thread's pipe, which would then never see EOF.
     */
    HANDLE inherit[3] = { wr, nul, errdup };
    const DWORD n_inherit = errdup ? 3 : 2;
    SIZE_T attr_size = 0;
    InitializeProcThreadAttributeList(NULL, 1, 0, &attr_size);
    LPPROC_THREAD_ATTRIBUTE_LIST attrs =
        (LPPROC_THREAD_ATTRIBUTE_LIST)vv_alloc(attr_size);
    bool attrs_ok = attrs &&
        InitializeProcThreadAttributeList(attrs, 1, 0, &attr_size) &&
        UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                  inherit, n_inherit * sizeof(HANDLE),
                                  NULL, NULL);

    STARTUPINFOEXW si;
    memset(&si, 0, sizeof(si));
    si.StartupInfo.cb = sizeof(si);
    si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    si.StartupInfo.hStdInput = nul;
    si.StartupInfo.hStdOutput = wr;
    si.StartupInfo.hStdError = errh;
    si.lpAttributeList = attrs;

    PROCESS_INFORMATION pi;
    memset(&pi, 0, sizeof(pi));
    /*
     * No console window of its own -- except when it writes to ours, which a
     * console handle only accepts from a process attached to that console.
     */
    const DWORD flags = EXTENDED_STARTUPINFO_PRESENT |
                        (errdup ? 0 : CREATE_NO_WINDOW);
    const BOOL started = attrs_ok &&
        CreateProcessW(NULL, cmd.p, NULL, NULL, TRUE, flags,
                       NULL, NULL, &si.StartupInfo, &pi);

    if (attrs) {
        if (attrs_ok) DeleteProcThreadAttributeList(attrs);
        vv_free(attrs);
    }
    vv_free(cmd.p);
    CloseHandle(wr);
    CloseHandle(nul);
    if (errdup) CloseHandle(errdup);
    if (!started) { CloseHandle(rd); return NULL; }
    CloseHandle(pi.hThread);

    const int fd = _open_osfhandle((intptr_t)rd, _O_RDONLY | _O_BINARY);
    FILE* f = fd >= 0 ? _fdopen(fd, "rb") : NULL;
    vv_child_t* c = f ? (vv_child_t*)vv_alloc(sizeof(*c)) : NULL;
    if (!c) {
        if (f) fclose(f);
        else if (fd >= 0) _close(fd);
        else CloseHandle(rd);
        TerminateProcess(pi.hProcess, 1);
        WaitForSingleObject(pi.hProcess, INFINITE);
        CloseHandle(pi.hProcess);
        return NULL;
    }
    c->process = pi.hProcess;
    c->out = f;
    *out = f;
    return c;
}

void vv_spawn_terminate(vv_child_t* c) {
    if (c) TerminateProcess(c->process, 1);
}

int vv_spawn_wait(vv_child_t* c, bool terminate) {
    if (!c) return -1;
    if (terminate) TerminateProcess(c->process, 1);
    if (c->out) fclose(c->out);
    WaitForSingleObject(c->process, INFINITE);
    DWORD code = 0;
    const int rc = GetExitCodeProcess(c->process, &code) ? (int)code : -1;
    CloseHandle(c->process);
    vv_free(c);
    return rc;
}

#else /* POSIX */

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

struct vv_child {
    pid_t pid;
    FILE* out;
};

/*
 * Both ends close-on-exec, so a child started by another thread at the same
 * moment cannot inherit this pipe (and keep its write end open). The dup2
 * in the file actions clears the flag on the child's own copy.
 */
static int cloexec_pipe(int fds[2]) {
#if defined(__linux__) || defined(__FreeBSD__) || defined(__NetBSD__) || \
    defined(__OpenBSD__)
    if (pipe2(fds, O_CLOEXEC) != 0) return -1;
#else
    if (pipe(fds) != 0) return -1;
    fcntl(fds[0], F_SETFD, FD_CLOEXEC);
    fcntl(fds[1], F_SETFD, FD_CLOEXEC);
#endif
    /*
     * If the process runs with stdin/stdout/stderr closed, pipe() may hand
     * back 0..2, and dup2(fd, fd) would leave the close-on-exec flag set on
     * the very descriptor the child needs. Move them out of the way.
     */
    for (int i = 0; i < 2; i++) {
        if (fds[i] <= 2) {
            const int moved = fcntl(fds[i], F_DUPFD_CLOEXEC, 3);
            if (moved < 0) { close(fds[0]); close(fds[1]); return -1; }
            close(fds[i]);
            fds[i] = moved;
        }
    }
    return 0;
}

vv_child_t* vv_spawn_read(const char* const argv[], vv_spawn_stderr_t err,
                          FILE** out) {
    if (!argv || !argv[0] || !argv[0][0] || !out) return NULL;
    *out = NULL;

    int fds[2];
    if (cloexec_pipe(fds) != 0) return NULL;

    posix_spawn_file_actions_t fa;
    if (posix_spawn_file_actions_init(&fa) != 0) {
        close(fds[0]); close(fds[1]);
        return NULL;
    }
    int rc = posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    if (!rc) rc = posix_spawn_file_actions_adddup2(&fa, fds[1], 1);
    if (!rc && err == VV_SPAWN_STDERR_MERGE)
        rc = posix_spawn_file_actions_adddup2(&fa, fds[1], 2);
    else if (!rc && err == VV_SPAWN_STDERR_NULL)
        rc = posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);

    pid_t pid = -1;
    if (!rc) rc = posix_spawnp(&pid, argv[0], &fa, NULL,
                               (char* const*)argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(fds[1]);
    if (rc != 0) { close(fds[0]); return NULL; }

    FILE* f = fdopen(fds[0], "r");
    vv_child_t* c = f ? (vv_child_t*)vv_alloc(sizeof(*c)) : NULL;
    if (!c) {
        if (f) fclose(f); else close(fds[0]);
        kill(pid, SIGTERM);
        while (waitpid(pid, NULL, 0) < 0 && errno == EINTR) {}
        return NULL;
    }
    c->pid = pid;
    c->out = f;
    *out = f;
    return c;
}

void vv_spawn_terminate(vv_child_t* c) {
    if (c) kill(c->pid, SIGTERM);
}

int vv_spawn_wait(vv_child_t* c, bool terminate) {
    if (!c) return -1;
    if (terminate) kill(c->pid, SIGTERM);
    if (c->out) fclose(c->out);
    int status = 0;
    pid_t r;
    do {
        r = waitpid(c->pid, &status, 0);
    } while (r < 0 && errno == EINTR);
    vv_free(c);
    if (r < 0) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
}

#endif
