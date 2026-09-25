#!/usr/bin/env bash
#
# Cut a release: pick the version, write it into the header and the
# changelog, commit, and tag. Pushing is left to the caller -- the Release
# workflow pushes the commit and the tag together, atomically.
#
#   scripts/release.sh auto            next version from the commits since
#                                      the last tag (the default)
#   scripts/release.sh patch|minor|major
#   scripts/release.sh 1.0.0-rc.1      exactly this version
#   scripts/release.sh --dry-run ...   print the plan, change nothing
#   scripts/release.sh --notes 0.2.0  print that version's changelog section
#
# What "auto" does, in Conventional Commits terms:
#   a `type!:` subject or a BREAKING CHANGE footer   -> major (minor below 1.0,
#                                                      where SemVer lets minor
#                                                      bumps break)
#   any `feat:`                                      -> minor
#   anything else                                    -> patch
# If the header already names a version above the last tag -- somebody bumped
# it by hand -- that version wins over anything smaller.
#
# The header holds X.Y.Z; a prerelease suffix lives only in the tag. CMake and
# CI both refuse a tag whose X.Y.Z disagrees with the header.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
HEADER=include/vibevoice/vibevoice.h
CHANGELOG=CHANGELOG.md
GPUSTACK=deploy/gpustack/vibevoice.yaml
REPO_URL="https://github.com/Ar4ikov/TurboQwen"
SEMVER='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z.-]+)?$'

die() { echo "release: $*" >&2; exit 1; }

header_version() {
    sed -n 's/^#define VV_VERSION_STRING "\(.*\)"$/\1/p' "$HEADER"
}

# vercmp A B -> prints -1, 0 or 1 comparing X.Y.Z numerically.
vercmp() {
    local IFS=.
    local -a a=($1) b=($2)
    for i in 0 1 2; do
        if (( ${a[i]} < ${b[i]} )); then echo -1; return; fi
        if (( ${a[i]} > ${b[i]} )); then echo 1; return; fi
    done
    echo 0
}

bump() {  # bump X.Y.Z kind
    local IFS=.
    local -a v=($1)
    case "$2" in
        major) echo "$(( v[0] + 1 )).0.0" ;;
        minor) echo "${v[0]}.$(( v[1] + 1 )).0" ;;
        patch) echo "${v[0]}.${v[1]}.$(( v[2] + 1 ))" ;;
        *) die "unknown bump '$2'" ;;
    esac
}

# The changelog section for one version, without its heading.
notes_for() {
    V="$1" awk '
        /^## / {
            if (on) exit
            h = $2; sub(/^\[/, "", h); sub(/\].*$/, "", h)
            if (h == ENVIRON["V"]) { on = 1; next }
        }
        on { print }
    ' "$CHANGELOG" | sed -e '/./,$!d' | sed -e ':a' -e '/^\n*$/{$d;N;ba' -e '}'
}

# The body of "## Unreleased", without its heading.
unreleased_notes() {
    awk '
        /^## / { if (on) exit; if ($2 == "Unreleased") { on = 1; next } }
        on { print }
    ' "$CHANGELOG" | sed -e '/./,$!d' | sed -e ':a' -e '/^\n*$/{$d;N;ba' -e '}'
}

DRY=0
if [ "${1:-}" = "--notes" ]; then
    [ -n "${2:-}" ] || die "--notes needs a version"
    notes_for "${2#v}"
    exit 0
fi
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
REQ="${1:-auto}"

[ -z "$(git status --porcelain --untracked-files=no)" ] || [ "$DRY" = 1 ] \
    || die "the working tree has uncommitted changes"

HDR="$(header_version)"
[ -n "$HDR" ] || die "no VV_VERSION_STRING in $HEADER"

LAST_TAG="$(git describe --tags --abbrev=0 --match 'v[0-9]*' 2>/dev/null || true)"
LAST="${LAST_TAG#v}"
LAST_REL="${LAST%%-*}"
RANGE="${LAST_TAG:+$LAST_TAG..}HEAD"
# Notes and the compare link run from the last *final* release, so that
# 0.3.0 lists everything since 0.2.0 and not just what followed 0.3.0-rc.1.
BASE_TAG="$(git describe --tags --abbrev=0 --match 'v[0-9]*' --exclude 'v*-*' 2>/dev/null || true)"
NOTES_RANGE="${BASE_TAG:+$BASE_TAG..}HEAD"

if [ -n "$LAST_TAG" ] && [ "$(git rev-list --count "$RANGE")" = 0 ]; then
    die "nothing to release: HEAD is $LAST_TAG"
fi

KIND=""
case "$REQ" in
    auto)
        subjects="$(git log --format='%s' "$RANGE")"
        bodies="$(git log --format='%b' "$RANGE")"
        if grep -Eq '^[a-z]+(\([^)]*\))?!:' <<<"$subjects" ||
           grep -Eq '^BREAKING[ -]CHANGE:' <<<"$bodies"; then
            KIND=major
        elif grep -Eq '^feat(\([^)]*\))?:' <<<"$subjects"; then
            KIND=minor
        else
            KIND=patch
        fi
        ;;
    patch|minor|major) KIND="$REQ" ;;
    *)
        NEXT="${REQ#v}"
        [[ "$NEXT" =~ $SEMVER ]] || die "'$REQ' is not a SemVer version"
        ;;
esac

if [ -n "$KIND" ]; then
    if [ -z "$LAST_TAG" ]; then
        # Nothing tagged yet: the first release is whatever the header says.
        NEXT="$HDR"
        KIND="first"
    else
        # Below 1.0 a breaking change is a minor bump.
        if [ "$KIND" = major ] && [ "${LAST_REL%%.*}" = 0 ]; then KIND=minor; fi
        if [ "$LAST" != "$LAST_REL" ]; then
            # Last tag was a prerelease: releasing finishes that version.
            NEXT="$LAST_REL"
        else
            NEXT="$(bump "$LAST_REL" "$KIND")"
        fi
        if [ "$(vercmp "$HDR" "$NEXT")" = 1 ]; then NEXT="$HDR"; fi
    fi
fi

NEXT_REL="${NEXT%%-*}"
PRE=0; [ "$NEXT" != "$NEXT_REL" ] && PRE=1
if [ -n "$LAST_TAG" ]; then
    c="$(vercmp "$NEXT_REL" "$LAST_REL")"
    if [ "$c" = -1 ] || { [ "$c" = 0 ] && [ "$LAST" = "$LAST_REL" ]; }; then
        die "$NEXT is not newer than $LAST_TAG"
    fi
fi
if git rev-parse -q --verify "refs/tags/v$NEXT" >/dev/null; then
    die "tag v$NEXT already exists"
fi

echo "release: last tag ${LAST_TAG:-none}, header $HDR, ${KIND:-explicit} -> $NEXT"
if [ "$DRY" = 1 ]; then exit 0; fi

# ─── Header ────────────────────────────────────────────────────────────────
IFS=. read -r MA MI PA <<<"$NEXT_REL"
sed -i \
    -e "s/^#define VV_VERSION_MAJOR .*/#define VV_VERSION_MAJOR $MA/" \
    -e "s/^#define VV_VERSION_MINOR .*/#define VV_VERSION_MINOR $MI/" \
    -e "s/^#define VV_VERSION_PATCH .*/#define VV_VERSION_PATCH $PA/" \
    -e "s/^#define VV_VERSION_STRING .*/#define VV_VERSION_STRING \"$NEXT_REL\"/" \
    "$HEADER"
[ "$(header_version)" = "$NEXT_REL" ] || die "failed to rewrite $HEADER"

# ─── Changelog ─────────────────────────────────────────────────────────────
# The Unreleased section becomes the release's section. When nobody wrote
# one, the commit subjects since the last tag stand in for it, so that a
# release never ships with no notes at all.
BODY="$(unreleased_notes)"
if [ -z "$BODY" ]; then
    BODY="$(git log --no-merges --format='- %s' "$NOTES_RANGE" \
            | grep -Ev '^- (chore|ci|docs|style|test)(\([^)]*\))?:' || true)"
    [ -n "$BODY" ] || BODY="- Maintenance release."
fi
DATE="$(date -u +%Y-%m-%d)"
if [ -n "$BASE_TAG" ]; then
    HEADING="## [$NEXT]($REPO_URL/compare/$BASE_TAG...v$NEXT) — $DATE"
else
    HEADING="## [$NEXT]($REPO_URL/releases/tag/v$NEXT) — $DATE"
fi
# Text goes to awk through the environment: -v would interpret backslashes.
export HEADING BODY
if [ "$PRE" = 1 ]; then
    # A prerelease gets its own section but leaves Unreleased in place: the
    # final release will carry the same notes.
    awk '
        !done && /^## / && $2 != "Unreleased" {
            print ENVIRON["HEADING"] "\n\n" ENVIRON["BODY"] "\n"; done = 1
        }
        { print }
        END { if (!done) print "\n" ENVIRON["HEADING"] "\n\n" ENVIRON["BODY"] }
    ' "$CHANGELOG" > "$CHANGELOG.tmp"
else
    awk '
        /^## Unreleased/ {
            print; print ""; print ENVIRON["HEADING"]; print ""
            print ENVIRON["BODY"]; print ""
            skip = 1; next
        }
        skip && /^## / { skip = 0 }
        skip { next }
        { print }
    ' "$CHANGELOG" > "$CHANGELOG.tmp"
fi
mv "$CHANGELOG.tmp" "$CHANGELOG"
grep -qF "$HEADING" "$CHANGELOG" || die "failed to rewrite $CHANGELOG"

# ─── GPUStack: offer the release line ──────────────────────────────────────
# One version_configs entry per X.Y line, added the first time the line is
# released; patches move the X.Y image tag on their own.
if [ "$PRE" = 0 ] && [ -f "$GPUSTACK" ]; then
    LINE="$MA.$MI"
    if ! grep -q "^  '$LINE':" "$GPUSTACK"; then
        awk -v line="$LINE" '
            { print }
            /^version_configs:/ { insec = 1; next }
            insec && /^    custom_framework:/ && !done {
                print "  # The " line " release line: newest v" line ".x, moves when a patch is tagged."
                print "  '\''" line "'\'':"
                print "    image_name: ghcr.io/ar4ikov/turboqwen:" line
                print "    entrypoint: /usr/local/bin/vv_cli"
                print "    custom_framework: cuda"
                done = 1
            }
        ' "$GPUSTACK" > "$GPUSTACK.tmp"
        mv "$GPUSTACK.tmp" "$GPUSTACK"
    fi
fi

git add "$HEADER" "$CHANGELOG" "$GPUSTACK"
git commit -q -m "chore: release $NEXT"
git tag -a "v$NEXT" -m "TurboQwen $NEXT" -m "$BODY"
echo "release: committed and tagged v$NEXT"
echo "version=$NEXT" >> "${GITHUB_OUTPUT:-/dev/null}"
echo "tag=v$NEXT" >> "${GITHUB_OUTPUT:-/dev/null}"
