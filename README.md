# hf-hat

This is the README file for the hf-hat program, code for computing HF-hat from a nice diagram.

## Usage: updated Python version

Consists of the files `hf_hat.py`, `example.py`, and `gf2.py` (as well as this readme and COPYING).

Requires Python 3 with `numpy` and `matplotlib` (`pip install numpy matplotlib`).

To run the example:

```
python3 example.py          # Run all examples (regenerates S237_0/1/2.pdf)
```

See `example.py` (especially), and the docstrings in `hf_hat.py`, for further documentation.

## Usage: legacy Sage version

Consists of the files `hf_hat.sage` and `example.sage` (as well as this readme and COPYING).

First, you need to install an appropriate version of [Sage](http://www.sagemath.org/). The code will not work with a modern version -- in particular, it expects a Python 2 version of Sage.

The main program is `hf-hat.sage`. Some detailed examples are worked out
in `example.sage`. See the files, particularly the file `example.sage`,
for further instructions.

Copy the `example.sage` file to some file `foo.sage`. Add your Heegaard
diagram to the file `foo.sage`. (Feel free to delete the older Heegaard
diagrams.)

Start `sage`. Then load `foo.sage`. (If you load `example.sage`, the program
will regenerate the three pdf files.)


## Credits, copyright, and license

Copyright (c) 2015-2026 by Sucharit Sarkar.
Contact: sucharit@math.ucla.edu

Python port created by Anthropic's Claude chatbot (Sonnet 4.6), with mild assistance by Robert Lipshitz.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
