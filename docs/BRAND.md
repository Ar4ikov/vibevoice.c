# Identity

<p align="center">
  <img src="../assets/brand/banner.svg" alt="vibevoice.c" width="760">
</p>

The runtime is a backend. Its identity is meant to survive being shrunk to a
favicon, printed in one colour, and pasted into a terminal-coloured README —
so it is a mark, a wordmark, three colours and nothing else.

## Where the mark comes from

VibeVoice's own mark is a **V whose right arm repeats**, the repeats cut
short like speed lines. This runtime keeps that resemblance, because it runs
that model and should look like it belongs to it, and then differs in three
deliberate ways:

- strokes end in **round caps**, where the original cuts them flat;
- there are **two repeats, not three**, which keeps the mark legible at 16 px;
- a **dot sits on the baseline** to the right. The dot is the period in
  `vibevoice.c`, and it is the one element allowed to carry the accent colour.

The name, the model and the original mark are Microsoft's. This one is the
runtime's own and carries no endorsement — see [Attribution](#attribution).

## Files

| file | use |
|---|---|
| [`assets/brand/lockup.svg`](../assets/brand/lockup.svg) | primary, on light surfaces |
| [`assets/brand/lockup-inverse.svg`](../assets/brand/lockup-inverse.svg) | primary, on dark surfaces |
| [`assets/brand/mark.svg`](../assets/brand/mark.svg) | the mark alone, light surfaces |
| [`assets/brand/mark-inverse.svg`](../assets/brand/mark-inverse.svg) | the mark alone, dark surfaces |
| [`assets/brand/banner.svg`](../assets/brand/banner.svg) | repository header; carries its own plate, so it needs no theme switch |
| [`assets/brand/icon.svg`](../assets/brand/icon.svg) | avatar, favicon, registry listing |

Every file is hand-written SVG with no external references, no embedded
fonts and no raster data. The wordmark is drawn as geometry, not set in a
typeface, so it renders identically on a machine with no fonts installed.

## Colour

| token | hex | where |
|---|---|---|
| Ink | `#0B0D12` | plates, and the mark on light surfaces |
| Paper | `#F7F8FA` | the mark on dark surfaces |
| Signal | `#4AE3B5` | the accent, **on ink only** |
| Signal deep | `#0E9E74` | the same accent where the surface is light |
| Slate | `#78859B` | supporting text, diagram strokes |

Two accent tints rather than one, because a mint that sings on near-black is
illegible on white. The rule is mechanical: **Signal on dark, Signal deep on
light**, and never accent on accent.

The accent is used for one thing at a time: the `.c`, the dot, a rule, a
single highlighted edge in a diagram. When everything is accented, nothing is.

## Geometry

Both parts share a 1:1.75 stroke-to-x-height ratio, which is what makes the
solid mark and the monoline wordmark read as one drawing.

**Mark** — `viewBox="0 0 66 48"`. Cap height 40, stroke 8, round caps and
joins. The V runs `4,4 → 19,44 → 34,4`; each repeat is parallel to the right
arm, offset 13.9 in x (a 5-unit gap between strokes) and cut at y=32 and
y=22. The dot is r=4 at `61.8,44`: its centre shares the x of the last
repeat's start and the y of the V's vertex, and its bottom edge lands on the
V's own bottom edge.

**Wordmark** — baseline y=100, x-height top y=60, ascender y=42, monoline
stroke 7. Every bowl is a circle of r=20, every terminal on `c` and `e` is
cut at 40° from the horizontal. Glyph boxes are separated by 9, except after
`c`, where 6 compensates for the open right side.

**Lockup** — the mark is scaled 1.354 so that it spans exactly the
wordmark's ink, ascender to baseline overshoot, and set 33 away from it:
half a cap height.

## Clear space and minimum size

Clear space on all four sides is **the height of the mark's dot** (8 mark
units, one stroke width). Nothing else goes inside it.

| asset | smallest that still reads |
|---|---|
| mark | 16 px tall |
| icon | 24 px |
| lockup | 120 px wide |
| banner | 480 px wide |

Below those, drop to the mark alone. The wordmark is never set below 120 px
wide; its counters close up.

## Badges

README badges are part of the identity, so they take the same two colours:
label side Ink, value side Signal deep. Build status keeps shields' own
red/green, because a badge that hides a failure is worse than a badge that
clashes.

```
https://img.shields.io/badge/<label>-<value>-0E9E74?labelColor=0B0D12
```

## Don't

- Recolour the mark to anything but Ink, Paper, or one flat colour.
- Put the accent on the strokes. It belongs to the dot and the `.c`.
- Set the wordmark in a font. It is geometry; the files are the source.
- Stretch, shear, rotate, outline, or add a gradient, a shadow or a glow.
- Place the lockup on a photograph or on a mid-tone. Use the banner, which
  brings its own plate.
- Imply that this is Microsoft's, or that Microsoft ships it.

## Attribution

VibeVoice, VibeVoice-ASR and the original V mark belong to Microsoft, and
the model is theirs under MIT. This runtime is an independent
reimplementation; its mark, wordmark and palette are its own and are MIT
along with the rest of the repository. Nothing here is endorsed by or
affiliated with Microsoft.
