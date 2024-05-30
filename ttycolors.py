import sys
from sys import stdin, stdout, stderr
import termios
import tty
import re
from contextlib import contextmanager
from fractions import Fraction
from typing import NewType, Callable, TextIO, Iterator
from collections import OrderedDict
from pathlib import Path
from functools import partial
from typing import assert_never


def eprint(*args, **kwargs):
    print(*args, file=stderr, **kwargs)


StdioNormal = TextIO
StdioRaw = NewType("StdioRaw", StdioNormal)


# documented in stackoverflow
@contextmanager
def stdio_raw(stdio: StdioNormal) -> Iterator[StdioRaw]:
    """Disable line editing and buffering on sys.std{in,out,err}."""
    fd = stdio.fileno()
    tattr = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd, termios.TCSANOW)
        yield StdioRaw(stdio)
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, tattr)


OSC = b"\x1b\x5d"
ST = b"\x1b\x5c"


class TtyControlError(RuntimeError):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def argstrs(self):
        yield from (f"{x!r}" for x in self.args)
        yield from (f"k={v!r}" for k, v in self.kwargs.items())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(self.argstrs())})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: " f"{'\n - '.join(self.argstrs())}"


def osc_write(request: bytes) -> None:
    r"""Writes `OSC || request || ST` to stdout. Does not care whether there's a reply."""
    [stdout.buffer.write(b) for b in (OSC, request, ST)]
    stdout.buffer.flush()


def osc_exchange(request: bytes, stdin=sys.stdin.buffer) -> bytes:
    r"""Writes `OSC || request || ST` to stdout, and tries to read `OSC || reply || ST (or '\a')` from stdin."""
    osc_write(request)
    # TODO timeout? via select?
    if (start := stdin.read(2)) != OSC:
        raise TtyControlError(
            "expected leading OSC",
            request=request,
            expected=OSC,
            received=start,
        )
    inbuf = b""
    # TODO better sanity check based on allowed control string bytes
    while True:
        if i := stdin.read(1):
            inbuf += i
        else:
            raise TtyControlError(
                "EOF during reply", request=request, partial_reply=inbuf
            )
        if (msg := inbuf.removesuffix(ST)) != inbuf:
            return msg
        elif (msg := inbuf.removesuffix(b"\a")) != inbuf:
            return msg
        else:
            continue


WHICH_OSC = r"([\d;]+?)"
RGBVAL = r"([0-9A-Fa-f]+)"
COLOR_REPLY_RE = re.compile(
    rf"^{WHICH_OSC};rgb:{RGBVAL}/{RGBVAL}/{RGBVAL}$".encode("ascii")
)
Component = Fraction
RgbColor = tuple[Component, Component, Component]


EXTRA_COLOR_OSCS = {
    "fg": b"10",
    "bg": b"11",
    "cursor": b"12",
}


def which_color_osc(ck: int | str) -> bytes:
    if isinstance(ck, int):
        return f"4;{ck:d}".encode("ascii")
    elif (x := EXTRA_COLOR_OSCS.get(ck)) is not None:
        return x
    else:
        raise ValueError("unsupported color ID", ck)


def write_color(colorkey: int | str, color: RgbColor) -> None:
    def strcomponent(frac) -> str:
        return f"{round(frac*0xffff):04x}"

    r, g, b = map(strcomponent, color)
    osc = which_color_osc(colorkey) + f";rgb:{r}/{g}/{b}".encode("ascii")
    osc_write(osc)


def read_color(color: int | str, stdin=None) -> RgbColor:
    # the numbered palette colors are requested as `4;<palette index>`
    which_osc = which_color_osc(color)
    request = which_osc + b";?"
    reply = osc_exchange(request)
    m = COLOR_REPLY_RE.match(reply)
    if not m or m.group(1) != which_osc:
        raise TtyControlError(
            "bad color query reply",
            dict(
                color=color,
                request=request,
                reply=reply,
                regex=COLOR_REPLY_RE,
                groups=None if m is None else m.groups(),
            ),
        )
    r, g, b = m.group(2, 3, 4)

    # xterm says each component is exactly 4, 8, 12, or 16 bits and is exactly
    # that many hex digits. it's probably safe to parse general sequences of hex digits.
    # fun math facts: 2**(4n) == 16**n and always ends in the digit 6, so
    # (16**n)-1 always ends in the digit 5, so fractions with denominators 0xf,
    # 0xff, 0xfff, 0xffff are very compatible.
    def component(x: bytes) -> Component:
        try:
            i = int(x, 16)
        except ValueError:
            raise TtyControlError("color component is not hex digits", x)
        # int allows a leading 0x, so subtle bug if not for the regex above
        # checking these are only hex digits.
        divisor = (16 ** len(x)) - 1
        return Fraction(i, divisor)

    return component(r), component(g), component(b)


def read_color_nothrow(*args, **kwargs) -> RgbColor | TtyControlError:
    try:
        return read_color(*args, **kwargs)
    except TtyControlError as e:
        return e


Palette = OrderedDict[int | str, RgbColor | None | TtyControlError]


def read_palette_tty() -> Palette:
    first_query = read_color_nothrow(0)
    if isinstance(first_query, TtyControlError):
        raise RuntimeError("first color query failed; is OSC 4 unsupported?")
    else:
        palette = Palette()
        palette[0] = first_query

        def colors() -> Iterator[int | str]:
            yield from range(1, 16)
            yield from EXTRA_COLOR_OSCS.keys()

        for color in colors():
            palette[color] = read_color_nothrow(color)
    return palette


def write_palette_tty(palette: Palette) -> None:
    for k, v in palette.items():
        if isinstance(v, tuple):
            write_color(k, v)


def default_format_output_color(c: RgbColor) -> str:
    # this just crunches it to the nearest 8-bit number, so
    # the output is always ranged 0-255.
    return " ".join(str(round(x * 255)) for x in c)


def default_format_output_palette(palette: Palette) -> Iterator[str]:
    for key, rgb in palette.items():
        match rgb:
            case None:
                pass
            case TtyControlError():
                yield f"# {key}:\ttty error: {rgb!r}"
            case tuple():
                yield f"{key}:\t{default_format_output_color(rgb)}"
            case never:
                assert_never(never)


def default_format_output(
    palette: Palette,
    writeln: Callable[[str], None] = print,
) -> None:
    [writeln(s) for s in default_format_output_palette(palette)]


def rgb_htmlcode(rgb: RgbColor) -> str:
    """Format a color as an 8-bit `#abcdef` hex code."""

    def hexcomp(x: Fraction) -> str:
        x8bit = round(x * 255)
        if x8bit > 255:
            raise ValueError("bad color component", x, rgb)
        return f"{x8bit:02x}"

    return "#" + "".join(map(hexcomp, rgb))


def output_alacritty(
    palette: Palette,
    writeln: Callable[[str], None] = print,
) -> None:
    def output_table(table_name, toml_names, pal_keys):
        writeln(f"[{table_name}]")
        for toml_name, key in zip(toml_names, pal_keys):
            output_line(toml_name, palette.get(key))
        writeln("")

    def output_line(name: str, val: RgbColor | TtyControlError | None):
        if val is None:
            writeln(f"# {name}: unset")
        elif isinstance(val, TtyControlError):
            writeln(f"# {name}: error {val!r}")
        else:
            writeln(f'{name} = "{rgb_htmlcode(val)}"')

    output_table("colors.primary", ("foreground", "background"), ("fg", "bg"))
    output_table("colors.cursor", ("cursor", "text"), ("cursor", "cursor_text"))
    ansi_names = ("black", "red", "green", "yellow", "blue", "magenta", "cyan", "white")
    output_table("colors.normal", ansi_names, range(0, 8))
    output_table("colors.bright", ansi_names, range(8, 16))


OUTPUT_FORMATS = {
    "default": default_format_output,
    "alacritty": output_alacritty,
}


def read_palette_default(f):
    def entries():
        for line in f.readlines():
            if line.lstrip().startswith("#"):
                pass
            else:
                lhs, rhs = line.split(":")

                def tryint(s: str) -> int | str:
                    try:
                        return int(s)
                    except ValueError:
                        return s

                colorkey = tryint(lhs)

                def tocolor(s: str) -> Fraction:
                    i = int(s)
                    assert i in range(0, 256)
                    return Fraction(i, 255)

                r, g, b = map(tocolor, rhs.split())
                yield colorkey, (r, g, b)

    return Palette(entries())


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="""Read or write the terminal emulator's color palette.""",
        # unwanted line wrapping:
        # The operation depends if one, both, or neither of `-i` and `-o` are given:
        # * `-i FILE`: read palette from FILE and set it as current terminal palette.
        # * `-o FILE`: read the current terminal palette and save it to FILE.
        # * neither flag: read the current terminal palette and write it to stdout.
        # * both flags: load a file and save it in another format.""",
    )
    p.add_argument(
        "-i",
        "--input-file",
        metavar="FILE",
        type=Path,
        help="""Read colors from FILE. By default, read colors from the
terminal attached to stdin and stdout. For now the only supported input
format is the unspecified default ttycolors format.""",
    )
    # TODO input formats

    p.add_argument(
        "-o",
        "--output-file",
        metavar="FILE",
        type=Path,
        help="""Write colors to FILE. By default, write colors to the terminal attached to stdin and stdout.""",
    )
    p.add_argument(
        "-O",
        "--output-format",
        # metavar="FORMAT",
        # ^ when choices are given, argparse replaces metavar with a literal
        # list of the choices. it won't show the choices anywhere else besides
        # the error message for an invalid value given.
        choices=OUTPUT_FORMATS.keys(),
        default="default",
        help="""Output the terminal palette in this format. The default format
is undefined and subject to change, but is intended to be parseable from
a shell script. Has no effect when the output is the terminal.""",
    )

    args = p.parse_args()
    outter = OUTPUT_FORMATS[args.output_format]
    match args.input_file:
        case None:
            with stdio_raw(stdin):
                palette = read_palette_tty()
        case Path() as path:
            with path.open("r") as f:
                palette = read_palette_default(f)
        case never:
            assert_never(never)

    match args.output_file:
        case None:
            if args.input_file is None:
                # don't query and then write the term emu to itself.
                # instead query and dump to stdout.
                outter(palette, print)
            else:
                write_palette_tty(palette)
        case Path() as path:
            with path.open("w") as f:
                writeln = partial(print, file=f)
                outter(palette, writeln)
        case never:
            assert_never(never)


if __name__ == "__main__":
    main()
