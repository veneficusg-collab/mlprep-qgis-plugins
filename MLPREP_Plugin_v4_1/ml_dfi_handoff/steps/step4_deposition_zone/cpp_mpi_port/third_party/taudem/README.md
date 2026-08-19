# Vendored TauDEM Compatibility Sources

This directory contains the small TauDEM source subset required to compile the
Step 4 C++/MPI executable:

- raster partition and neighborhood templates
- GeoTIFF I/O helpers
- shared utility code

The subset was copied from the TauDEM sources previously used to build Step 4
so the workspace no longer depends on a separate TauDEM checkout on one
developer's desktop. The retained `license.txt` applies to these files.

One portability-only correction was made in `src/linearpart.h`: a diagnostic
`printf` uses a format compatible with the unsigned 64-bit partition index
type. No routing or raster-processing logic was changed.
