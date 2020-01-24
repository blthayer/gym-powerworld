# cases
This directory is used to store PowerWorld cases.

## ieee_14
IEEE 14-bus test case downloaded from [here](https://electricgrids.engr.tamu.edu/electric-grid-test-cases/ieee-14-bus-system/)
on 2019-10-15.

NOTE: There is an additional .pwb file, entitled
"IEEE 14 bus condensers.PWB" in which the MW limits (low and high) have
been set to 0 for the generators at buses 3, 6, and 8. This represents
how the original case was intended to be, and also matches what the 
GridMind folks did in their work (they didn't dispatch real power from
those generators in their scenarios).