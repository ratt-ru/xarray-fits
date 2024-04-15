from collections import Counter
from collections.abc import Sequence, Mapping
from xarrayfits.typing import ChunksType, HduType


def promote_chunks(chunks: ChunksType, nhdus: int) -> Sequence[Mapping[str, int]]:
    pchunks: Sequence[Mapping[str, int]]

    if chunks is None:
        pchunks = [{}] * nhdus
    # Promote to list in case of single dict
    elif isinstance(chunks, Mapping):
        pchunks = [chunks]
    elif isinstance(chunks, Sequence):
        if not all(isinstance(c, Mapping) for c in chunks):
            raise TypeError(f"{chunks} should be an Iterable of Mapping[str, int]")

        # Use default chunking if chunks aren't specified for an hdu
        if len(chunks) < nhdus:
            pchunks = list(chunks) + [{}] * (nhdus - len(chunks))
        else:
            pchunks = chunks
    else:
        raise TypeError(f"Invalid chunks {chunks}")

    return pchunks


def promote_hdus(hdus: HduType, nhdus: int) -> Mapping[int, str]:
    """Promotes hdus specified in terms of ``HduType`` to a ``Mapping[int, str]``"""
    type_err_msg = (
        f"hdus must a int, str, "
        f"Sequence[int], Sequence[str], "
        f"or a Mapping[int, str]"
    )

    phdus: Mapping[int, str]

    # Take all hdus if None specified
    if hdus is None:
        if nhdus == 1:
            phdus = {0: "hdu"}
        else:
            phdus = {i: f"hdu{i}" for i in range(nhdus)}
    # promote to list in case of single integer or string
    elif isinstance(hdus, int):
        phdus = {hdus: "hdu"}
    elif isinstance(hdus, str):
        phdus = {0: hdus}
    elif isinstance(hdus, Sequence):
        all_ints = all(isinstance(k, int) for k in hdus)
        all_str = all(isinstance(k, str) for k in hdus)

        if all_ints:
            if not all(i < nhdus for i in hdus):
                raise ValueError(
                    f"There are {nhdus} present, but "
                    f"{hdus} references non-existent hdus"
                )

            if len(hdus) == 1:
                phdus = {hdus[0]: "hdu"}
            else:
                phdus = {k: f"hdu{k}" for k in hdus}
        elif all_str:
            if not len(hdus) < nhdus:
                raise ValueError(
                    f"There are {nhdus} present, but "
                    f"{hdus} references more than "
                    f"are present"
                )
            if len(hdus) == 1:
                phdus = {0: hdus[0]}
            else:
                phdus = {i: k for i, k in enumerate(hdus)}
        else:
            raise TypeError(type_err_msg)
    elif isinstance(hdus, Mapping):
        phdus = hdus
    else:
        raise TypeError(type_err_msg)

    if not all(isinstance(i, int) and i < nhdus for i in phdus.keys()):
        raise ValueError(f"{phdus} keys must be integers")
    if any(v > 1 for v in Counter(phdus.values()).values()):
        raise ValueError(f"{phdus} values must be unique strings")

    return phdus
