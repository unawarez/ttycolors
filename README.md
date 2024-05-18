# ttycolors

This is half documentation of how to read current palette values from a terminal emulator, and half utility script for dumping and importing terminal themes.
Moral of the story: palette reading and writing happens via some [XTerm OSCs](https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h3-Operating-System-Commands).
Ideally this script would dump or load palettes in a handful of formats.

## Can do

* Read the current terminal palette, colors 0-15 plus a couple specials
* Output the palette in either an unspecified default format or in Alacritty
  config TOML format

## TODO

* Handle tty comm issues at all, namely timeout instead of hanging forever
* *Read* palettes and *write* the current terminal colors
* More formats
