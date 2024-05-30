# ttycolors

This is half documentation of how to read current palette values from a terminal emulator, and half utility script for dumping and importing terminal themes.
Moral of the story: palette reading and writing happens via some [XTerm OSCs](https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h3-Operating-System-Commands).
Ideally this script would dump or load palettes in a handful of formats.

## Features

* Read the current terminal palette, colors 0-15 plus a couple specials, and print or save it in one of two formats
* Read a palette file in a dumb unspecified format and set it as the current terminal palette
* Convert palette files from the dumb format to Alacritty config format
* Hang forever if any tty communication problem happens
