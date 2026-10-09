# xarray-fits

Presents FITS images as Image Datasets that conform to the MSv4 image schema.

## Language

**FITS Image**:
A FITS file whose primary HDU holds an n-dimensional image with a celestial WCS. This is the input.
_Avoid_: FITS cube, HDU (when you mean the whole file)

**Image Dataset**:
A single dataset that conforms to the MSv4 image schema and gathers one or more Images that share time, frequency, polarization and sky coordinates.
_Avoid_: MSv4 image, image xds, sky image

**Image Schema**:
The versioned MSv4 specification for Image Datasets: their dimensions, coordinates, data variables and attributes.
_Avoid_: MSv4 spec (when you mean the image part specifically)

**Role**:
What an Image represents inside an Image Dataset, such as sky, residual, model, point spread function, primary beam or mask.
_Avoid_: image type, kind

**Data Group**:
A named set of Roles inside an Image Dataset that belong together, for example a sky image with its flag and beam fit parameters.
_Avoid_: group, image set
