from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# Paths (relative)
HERE = Path(__file__).resolve()
OPTIONS_DIR = HERE.parents[1]
OUT_DIR = OPTIONS_DIR / "data" / "processed"

CHAIN_CHUNKS_DIR = OUT_DIR / "chain_chunks"
IV_CHUNKS_DIR = OUT_DIR / "iv_chunks"

CHAIN_OUT = OUT_DIR / "spy_chain_clean.parquet"
IV_OUT = OUT_DIR / "spy_iv_points.parquet"


def stream_concat_parquets(input_files: list[Path], output_file: Path) -> None:
    if not input_files:
        raise FileNotFoundError(f"No input parquet files found for {output_file.name}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_file.with_suffix(output_file.suffix + ".tmp")

    writer = None
    try:
        for i, f in enumerate(input_files, start=1):
            table = pq.read_table(f)

            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="snappy")

            writer.write_table(table)

            if i % 25 == 0:
                print(f"Wrote {i}/{len(input_files)} parts into {output_file.name}...")

    finally:
        if writer is not None:
            writer.close()

    tmp.replace(output_file)
    print(f"✅ Wrote {output_file} from {len(input_files)} parts")


def main():
    chain_files = sorted(CHAIN_CHUNKS_DIR.glob("chain_*.parquet"))
    iv_files = sorted(IV_CHUNKS_DIR.glob("iv_*.parquet"))

    print(f"Found chain parts: {len(chain_files)}")
    print(f"Found iv parts:    {len(iv_files)}")

    stream_concat_parquets(chain_files, CHAIN_OUT)
    stream_concat_parquets(iv_files, IV_OUT)


if __name__ == "__main__":
    main()
