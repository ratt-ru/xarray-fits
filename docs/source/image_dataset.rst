Image Datasets
==============

An Image Dataset conforms to version ``0.0.2`` of the MSv4 Image Schema.
This page describes how FITS Images map onto it.

Dimensions and coordinates
--------------------------

``time``
  A single observation date in MJD days, from ``DATE-OBS`` or ``MJD-OBS``
  in the ``TIMESYS`` scale. Without a date it is MJD 0.0, with a warning.

``frequency``
  Channel frequencies in Hz, from a ``FREQ``, ``VOPT``, ``VRAD`` or
  ``FELO`` axis. Its attributes hold the spectral frame, the rest,
  reference frequency and channel width. Without a spectral axis there is
  a single channel at 1.415 GHz.

``polarization``
  Labels in canonical order, ``I, Q, U, V``, ``RR, RL, LR, LL`` and
  ``XX, XY, YX, YY``. FITS stores correlations as ``RR, LL, RL, LR``,
  and the planes are reordered lazily. Without a ``STOKES`` axis it is
  ``I``.

``velocity``
  Velocities in m/s on the ``frequency`` dimension, when the rest
  frequency is known.

``l`` and ``m``
  Projection plane coordinates in radians. ``l`` keeps the sign of
  ``CDELT``, so it usually decreases.

``right_ascension`` and ``declination``
  Sky coordinates of each pixel in radians, on ``(l, m)``, computed
  lazily through the full WCS.

``beam_params_label``
  ``major``, ``minor`` and ``pa``.

Data variables
--------------

Each Image is a data variable named by its Role, for example ``SKY``,
``POINT_SPREAD_FUNCTION``, ``PRIMARY_BEAM``, ``SKY_RESIDUAL`` or
``VISIBILITY_NORMALIZATION``. The sum of weights has no ``l`` and ``m``.

``FLAG_<ROLE>``
  A lazy flag of the NaN pixels of every floating point Image. It exists
  whether or not the Image has NaNs, so that opening never scans the
  pixels. Integer Images have no flag.

``BEAM_FIT_PARAMS_<ROLE>``
  Restoring beams in radians, from ``BMAJ``, ``BMIN`` and ``BPA`` or a
  ``BEAMS`` table of per-plane beams, for sky Images and the point spread
  function.

Attributes
----------

``data_groups``
  Each sky Image has a Data Group, ``base`` for ``SKY`` and ``residual``
  for ``SKY_RESIDUAL``, for example, holding it with its flag and beams.
  The other Images join every Data Group.

``coordinate_system_info``
  The projection, reference direction and frame, native pole, PC matrix
  and projection parameters of the celestial axes.

Image attributes hold ``units`` (``BUNIT``), ``sub_type`` (``BTYPE``),
``object_name``, ``observer``, ``obsdate``, ``pointing_center``,
``telescope`` and the remaining header cards as ``user``.

Unsupported FITS Images
-----------------------

Compressed HDUs, scaled pixels (``BSCALE`` or ``BZERO``), ``CDi_j``
matrices, galactic and aperture axes and other spectral axes raise
errors. Extension HDUs other than ``BEAMS`` are ignored with a warning.
