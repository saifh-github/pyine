import asyncio
import pathlib

import dotenv

import pyine.utils.async_processing

if __name__ == "__main__":
    dotenv.load_dotenv()
    asyncio.run(
        pyine.utils.async_processing.reprocess_code_samples(
            output_dir_path=pathlib.Path("./data/2025-03-26-v01/"),
        ),
    )
