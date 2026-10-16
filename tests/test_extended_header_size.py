# Copyright (c) 2016, Science and Technology Facilities Council
# This software is distributed under a BSD licence. See LICENSE.txt.

"""Byte-level regressions for extended-header lengths at write time."""

import bz2
import contextlib
import gzip
import io
import itertools
import struct
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from mrcfile.bzip2mrcfile import Bzip2MrcFile
from mrcfile.gzipmrcfile import GzipMrcFile
from mrcfile.mrcfile import MrcFile
from mrcfile.mrcinterpreter import MrcInterpreter
from mrcfile.mrcmemmap import MrcMemmap


class ExtendedHeaderSizeTest(unittest.TestCase):
    @contextlib.contextmanager
    def _assert_size_warning(self, old_size, actual_size):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield
        if old_size == actual_size:
            self.assertEqual(caught, [])
        else:
            self.assertEqual(len(caught), 1)
            self.assertIs(caught[0].category, RuntimeWarning)
            self.assertEqual(
                str(caught[0].message),
                "Extended header size does not match header.nsymbt. "
                f"Updating header.nsymbt from {old_size} to {actual_size}.",
            )

    def test_file_writers(self):
        writers = (MrcFile, MrcMemmap, GzipMrcFile, Bzip2MrcFile)
        for writer, byte_order, metadata_size, stale_size in itertools.product(
            writers, ("<", ">"), (0, 16), (0, 8, 16, 160)
        ):
            with (
                self.subTest(
                    writer=writer.__name__,
                    byte_order=byte_order,
                    metadata_size=metadata_size,
                    stale_size=stale_size,
                ),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "volume.mrc"
                data = np.arange(60, dtype=byte_order + "i2").reshape(3, 4, 5)
                metadata = np.arange(metadata_size // 2, dtype="i2").reshape(
                    2, metadata_size // 4
                )
                with self._assert_size_warning(0, 0), writer(path, mode="w+") as mrc:
                    mrc.set_data(data)
                    mrc.set_extended_header(metadata)
                    mrc.header.nsymbt = stale_size
                    header_before = mrc.header.tobytes()
                    with self._assert_size_warning(stale_size, metadata.nbytes):
                        mrc.flush()
                    assert mrc.header.nsymbt == metadata.nbytes
                    header_after = mrc.header.tobytes()
                    assert header_after[:92] == header_before[:92]
                    assert header_after[96:] == header_before[96:]
                    # Repeated flush and close must not warn again.
                    with self._assert_size_warning(metadata.nbytes, metadata.nbytes):
                        mrc.flush()
                raw = path.read_bytes()
                if writer is GzipMrcFile:
                    raw = gzip.decompress(raw)
                elif writer is Bzip2MrcFile:
                    raw = bz2.decompress(raw)
                assert struct.unpack_from(byte_order + "i", raw, 92)[0] == (
                    metadata.nbytes
                )
                assert len(raw) == 1024 + metadata.nbytes + data.nbytes
                assert raw[1024 : 1024 + metadata.nbytes] == metadata.tobytes()
                assert raw[1024 + metadata.nbytes :] == data.tobytes()

    def test_stream_writer(self):
        for byte_order, metadata_size in itertools.product(("<", ">"), (0, 16)):
            with self.subTest(byte_order=byte_order, metadata_size=metadata_size):
                stream = io.BytesIO()
                data = np.arange(12, dtype=byte_order + "f4").reshape(3, 4)
                with self._assert_size_warning(0, 0), MrcInterpreter() as mrc:
                    mrc._create_default_attributes()
                    mrc._iostream = stream
                    mrc.set_data(data)
                    mrc.set_extended_header(np.arange(metadata_size, dtype="u1"))
                    mrc.header.nsymbt = 160
                    with self._assert_size_warning(160, metadata_size):
                        mrc.flush()
                    with self._assert_size_warning(metadata_size, metadata_size):
                        mrc.flush()
                    raw = stream.getvalue()
                    assert struct.unpack_from(byte_order + "i", raw, 92)[0] == (
                        metadata_size
                    )
                    assert raw[1024 + metadata_size :] == data.tobytes()
                stream.seek(0)
                with MrcInterpreter(stream) as restored:
                    np.testing.assert_array_equal(restored.data, data)

    def test_copying_main_header_without_extended_header(self):
        writers = (MrcFile, MrcMemmap, GzipMrcFile, Bzip2MrcFile)
        for writer, byte_order, timing in itertools.product(
            writers, ("<", ">"), ("before", "after", "replacement")
        ):
            with (
                self.subTest(
                    writer=writer.__name__, byte_order=byte_order, timing=timing
                ),
                tempfile.TemporaryDirectory() as directory,
            ):
                source = Path(directory) / "source.mrc"
                target = Path(directory) / "target.mrc"
                data = np.arange(1, 13, dtype=byte_order + "i2").reshape(3, 4)
                with MrcFile(source, mode="w+") as mrc:
                    mrc.set_data(data)
                    mrc.set_extended_header(np.zeros(160, dtype="u1"))
                    source_header = mrc.header.copy()
                with self._assert_size_warning(0, 0), writer(target, mode="w+") as mrc:
                    if timing == "after":
                        mrc.set_data(data)
                    elif timing == "replacement":
                        mrc.set_data(np.zeros_like(data))
                    mrc.header[...] = source_header
                    if timing != "after":
                        old_size = 160 if writer is MrcMemmap else 0
                        with self._assert_size_warning(old_size, 0):
                            mrc.set_data(data)
                    if writer is MrcMemmap:
                        assert mrc.data.offset == 1024
                    old_size = 0 if writer is MrcMemmap and timing != "after" else 160
                    with self._assert_size_warning(old_size, 0):
                        mrc.flush()
                    with self._assert_size_warning(0, 0):
                        mrc.flush()
                raw = target.read_bytes()
                if writer is GzipMrcFile:
                    raw = gzip.decompress(raw)
                elif writer is Bzip2MrcFile:
                    raw = bz2.decompress(raw)
                assert struct.unpack_from(byte_order + "i", raw, 92)[0] == 0
                assert raw[1024:] == data.tobytes()
                reader = MrcFile if writer is MrcMemmap else writer
                with reader(target) as restored:
                    np.testing.assert_array_equal(restored.data, data)

    def test_read_only_flush_does_not_change_header_or_file(self):
        for writer in (MrcFile, MrcMemmap, GzipMrcFile, Bzip2MrcFile):
            with (
                self.subTest(writer=writer.__name__),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "volume.mrc"
                with writer(path, mode="w+") as mrc:
                    mrc.set_data(np.arange(12, dtype="i2").reshape(3, 4))
                original = path.read_bytes()
                with self._assert_size_warning(0, 0), writer(path) as mrc:
                    mrc.header.flags.writeable = True
                    mrc.header.nsymbt = 160
                    mrc.header.flags.writeable = False
                    header_before = mrc.header.tobytes()
                    mrc.flush()
                    mrc.flush()
                    assert mrc.header.nsymbt == 160
                    assert mrc.header.tobytes() == header_before
                assert path.read_bytes() == original
