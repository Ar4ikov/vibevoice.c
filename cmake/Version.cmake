# Version.cmake -- what version this build is.
#
# One release number lives in include/vibevoice/vibevoice.h
# (VV_VERSION_STRING). Every build derives a full SemVer 2.0 string from it
# and from git, by the same rule everywhere -- here, in CI and in the image:
#
#   exactly on tag v0.2.0, clean          0.2.0
#   5 commits past v0.2.0                 0.2.1-dev.5+g1a2b3c4
#   header bumped to 0.3.0, not tagged    0.3.0-dev.5+g1a2b3c4
#   5 commits past v0.3.0-rc.1            0.3.0-rc.1.dev.5+g1a2b3c4
#   uncommitted changes                   ...+g1a2b3c4.dirty
#   no git at all                         0.2.0, revision "unknown"
#
# A dev build is a prerelease of the *next* version, so it sorts after the
# release it came from and before the one it is heading for, which is what
# SemVer precedence needs for "newer" to mean newer. The part after `-dev.`
# counts commits since the tag and only grows on a branch that is not
# rewritten, so on master every build has a number of its own.
#
# Callers can pin the result with -DVV_VERSION_FULL=... -DVV_REVISION=...;
# the container build does, because its context has no .git.
#
# Used three ways:
#   include(cmake/Version.cmake); vv_compute_version()   from CMakeLists.txt
#   cmake -DVV_OUT=<header> -P cmake/Version.cmake        stamp a header at
#                                                         build time
#   cmake -P cmake/Version.cmake                          print key=value
#                                                         lines (for CI)

set(VV_SEMVER_REGEX
    "^(0|[1-9][0-9]*)[.](0|[1-9][0-9]*)[.](0|[1-9][0-9]*)(-[0-9A-Za-z.-]+)?([+][0-9A-Za-z.-]+)?$")

# The release number out of the header, "X.Y.Z".
function(vv_header_version source_dir out_var)
    file(READ "${source_dir}/include/vibevoice/vibevoice.h" _hdr)
    string(REGEX MATCH "VV_VERSION_STRING[ \t]+\"([0-9]+[.][0-9]+[.][0-9]+)\"" _m "${_hdr}")
    if(NOT CMAKE_MATCH_1)
        message(FATAL_ERROR "No VV_VERSION_STRING in include/vibevoice/vibevoice.h")
    endif()
    set(_v "${CMAKE_MATCH_1}")
    # The three numeric macros have to agree with the string.
    foreach(_part MAJOR MINOR PATCH)
        string(REGEX MATCH "#define VV_VERSION_${_part}[ \t]+([0-9]+)" _m "${_hdr}")
        if(NOT CMAKE_MATCH_COUNT EQUAL 1)
            message(FATAL_ERROR "No VV_VERSION_${_part} in include/vibevoice/vibevoice.h")
        endif()
        list(APPEND _nums "${CMAKE_MATCH_1}")
    endforeach()
    list(JOIN _nums "." _joined)
    if(NOT _joined STREQUAL _v)
        message(FATAL_ERROR "vibevoice.h: VV_VERSION_STRING ${_v} disagrees with "
                            "MAJOR.MINOR.PATCH ${_joined}")
    endif()
    set(${out_var} "${_v}" PARENT_SCOPE)
endfunction()

# Sets VV_VERSION (header), VV_VERSION_FULL (this build) and VV_REVISION
# (short commit, "-dirty" when the tree has changes, or "unknown").
function(vv_compute_version source_dir)
    vv_header_version("${source_dir}" _release)
    set(_full "${VV_VERSION_FULL}")
    set(_rev  "${VV_REVISION}")

    if(NOT _full)
        find_package(Git QUIET)
        if(GIT_FOUND AND EXISTS "${source_dir}/.git")
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" describe --tags --long --match "v[0-9]*"
                        --abbrev=7 --dirty=-dirty
                WORKING_DIRECTORY "${source_dir}"
                OUTPUT_VARIABLE _desc OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _rc ERROR_QUIET)
            set(_dirty FALSE)
            if(_rc EQUAL 0 AND _desc MATCHES "^v(.+)-([0-9]+)-g([0-9a-f]+)(-dirty)?$")
                set(_tag  "${CMAKE_MATCH_1}")
                set(_n    "${CMAKE_MATCH_2}")
                set(_sha  "${CMAKE_MATCH_3}")
                if(CMAKE_MATCH_4)
                    set(_dirty TRUE)
                endif()
            else()
                # No release tag reachable yet: count from the first commit.
                set(_tag "")
                execute_process(
                    COMMAND "${GIT_EXECUTABLE}" rev-list --count HEAD
                    WORKING_DIRECTORY "${source_dir}"
                    OUTPUT_VARIABLE _n OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
                execute_process(
                    COMMAND "${GIT_EXECUTABLE}" describe --always --abbrev=7 --dirty=-dirty
                    WORKING_DIRECTORY "${source_dir}"
                    OUTPUT_VARIABLE _sha OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
                if(_sha MATCHES "^([0-9a-f]+)(-dirty)?$")
                    set(_sha "${CMAKE_MATCH_1}")
                    if(CMAKE_MATCH_2)
                        set(_dirty TRUE)
                    endif()
                else()
                    set(_sha "")
                endif()
            endif()

            if(_sha)
                set(_rev "${_sha}")
                if(_dirty)
                    string(APPEND _rev "-dirty")
                endif()

                string(REGEX REPLACE "-.*$" "" _tag_release "${_tag}")
                if(_tag AND _n EQUAL 0 AND NOT _dirty)
                    # Checked out exactly at a release tag.
                    set(_full "${_tag}")
                else()
                    if(NOT _tag OR _release VERSION_GREATER _tag_release)
                        # The header was bumped ahead of the last tag.
                        set(_full "${_release}-dev.${_n}")
                    elseif(NOT _tag STREQUAL _tag_release AND _tag_release VERSION_EQUAL _release)
                        # Past a prerelease tag of the version in the header.
                        set(_full "${_tag}.dev.${_n}")
                    else()
                        string(REPLACE "." ";" _p "${_tag_release}")
                        list(GET _p 0 _maj)
                        list(GET _p 1 _min)
                        list(GET _p 2 _pat)
                        math(EXPR _pat "${_pat} + 1")
                        set(_full "${_maj}.${_min}.${_pat}-dev.${_n}")
                    endif()
                    string(APPEND _full "+g${_sha}")
                    if(_dirty)
                        string(APPEND _full ".dirty")
                    endif()
                endif()
            endif()
        endif()
    endif()

    if(NOT _full)
        set(_full "${_release}")
    endif()
    if(NOT _rev)
        set(_rev "unknown")
    endif()
    if(NOT _full MATCHES "${VV_SEMVER_REGEX}")
        message(FATAL_ERROR "'${_full}' is not a SemVer 2.0 version")
    endif()
    # Whatever the build calls itself has to be a build of the header's
    # release line or of the one after it, never of something unrelated.
    string(REGEX REPLACE "[-+].*$" "" _full_release "${_full}")
    if(_full_release VERSION_LESS _release)
        message(FATAL_ERROR "version ${_full} is older than the header's ${_release}")
    endif()

    set(VV_VERSION      "${_release}" PARENT_SCOPE)
    set(VV_VERSION_FULL "${_full}"    PARENT_SCOPE)
    set(VV_REVISION     "${_rev}"     PARENT_SCOPE)
endfunction()

# ─── Script mode ────────────────────────────────────────────────────────────
if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
    if(NOT VV_SOURCE_DIR)
        get_filename_component(VV_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
    endif()
    vv_compute_version("${VV_SOURCE_DIR}")

    if(VV_OUT)
        # Stamp the header, touching it only when the text changes so that
        # an unchanged version recompiles nothing.
        set(_text "/* Generated by cmake/Version.cmake -- do not edit. */\n"
                  "#define VV_VERSION_FULL \"${VV_VERSION_FULL}\"\n"
                  "#define VV_REVISION \"${VV_REVISION}\"\n")
        string(CONCAT _text ${_text})
        set(_old "")
        if(EXISTS "${VV_OUT}")
            file(READ "${VV_OUT}" _old)
        endif()
        if(NOT _old STREQUAL _text)
            file(WRITE "${VV_OUT}" "${_text}")
        endif()
    else()
        # key=value, one per line, for $GITHUB_OUTPUT and shell scripts.
        string(REGEX REPLACE "[+].*$" "" _image "${VV_VERSION_FULL}")
        if(VV_VERSION_FULL MATCHES "^[0-9.]+-")
            set(_pre true)
        else()
            set(_pre false)
        endif()
        execute_process(COMMAND "${CMAKE_COMMAND}" -E echo "version=${VV_VERSION_FULL}")
        execute_process(COMMAND "${CMAKE_COMMAND}" -E echo "release=${VV_VERSION}")
        execute_process(COMMAND "${CMAKE_COMMAND}" -E echo "revision=${VV_REVISION}")
        execute_process(COMMAND "${CMAKE_COMMAND}" -E echo "image=${_image}")
        execute_process(COMMAND "${CMAKE_COMMAND}" -E echo "prerelease=${_pre}")
    endif()
endif()
