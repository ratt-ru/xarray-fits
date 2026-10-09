"""Constants of the MSv4 Image Schema"""

#: Version of the Image Schema that Image Datasets conform to
IMAGE_SCHEMA_VERSION = "0.0.2"

#: Value of the ``type`` attribute of an Image Dataset
IMAGE_DATASET_TYPE = "image_dataset"

#: Dimensions of a sky-plane image, in order
SKY_DIMS = ("time", "frequency", "polarization", "l", "m")

#: Labels of the ``beam_params_label`` coordinate
BEAM_PARAMS_LABELS = ("major", "minor", "pa")

#: Notes describing the ``l`` and ``m`` coordinates
L_M_NOTES = {
  "l": "l is the projection plane coordinate towards the east, measured from "
  "the reference direction: l = x*cdelt, where x is the pixel offset from the "
  "reference pixel. For the SIN projection without projection parameters it "
  "is the direction cosine l of AIPS Memo #27, Section III.",
  "m": "m is the projection plane coordinate towards the north, measured from "
  "the reference direction: m = y*cdelt, where y is the pixel offset from the "
  "reference pixel. For the SIN projection without projection parameters it "
  "is the direction cosine m of AIPS Memo #27, Section III.",
}

#: Canonical order of the polarization axis, which keeps the correlations
#: of a pair of feeds in Jones matrix order
CANONICAL_POLARIZATION_ORDER = (
  "I",
  "Q",
  "U",
  "V",
  "RR",
  "RL",
  "LR",
  "LL",
  "XX",
  "XY",
  "YX",
  "YY",
  "RX",
  "RY",
  "LX",
  "LY",
  "XR",
  "XL",
  "YR",
  "YL",
  "PP",
  "PQ",
  "QP",
  "QQ",
)
