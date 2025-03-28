import asyncio
import pathlib

import dotenv

import src.utils.async_processing


if __name__ == "__main__":
    dotenv.load_dotenv()
    asyncio.run(
        src.utils.async_processing.reprocess_code_samples(
            output_dir_path=pathlib.Path("./data/2025-03-26-v01/"),
        ),
    )
