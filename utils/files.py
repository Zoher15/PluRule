"""
File processing utilities for Reddit data pipeline.

Provides shared functions for reading/writing compressed files,
filtering by date, and parallel processing.
"""

import os
import zstandard
import json
import time
import multiprocessing
from typing import Callable, List, Tuple, Dict, Any




# JSON parsing with fallback
try:
    import orjson
    json_loads = orjson.loads
    json_dumps = lambda obj: orjson.dumps(obj).decode('utf-8')
    json_dumps_pretty = lambda obj: orjson.dumps(obj, option=orjson.OPT_INDENT_2).decode('utf-8')
except ImportError:
    try:
        import ujson
        json_loads = ujson.loads
        json_dumps = ujson.dumps
        json_dumps_pretty = lambda obj: ujson.dumps(obj, indent=2)
    except ImportError:
        json_loads = json.loads
        json_dumps = json.dumps
        json_dumps_pretty = lambda obj: json.dumps(obj, indent=2)


def read_and_decode(reader, chunk_size=2**24, max_window_size=(2**29)*2, previous_chunk=None, bytes_read=0):
    """Recursively decompress and decode chunks with error handling."""
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size

    if previous_chunk is not None:
        chunk = previous_chunk + chunk

    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def process_zst_file(input_file: str, output_file: str, line_processor: Callable[[str], bool],
                     progress_interval: int = 10_000_000, logger=None) -> Dict[str, int]:
    """
    Process a compressed file line by line with single output.

    Args:
        input_file: Path to input .zst file
        output_file: Path to output .zst file
        line_processor: Function that takes a line and returns True if it should be kept
        progress_interval: Log progress every N lines
        logger: Optional logger for progress messages (if None, uses print)

    Returns:
        Dictionary with processing statistics
    """
    def single_output_processor(line: str, processors: Dict) -> Dict[str, Any]:
        """Simple wrapper for single output compatibility."""
        if line_processor(line):
            return {'matched': True, 'output_files': [output_file], 'data': line}
        return {'matched': False}

    return process_zst_file_multi(input_file, single_output_processor, {}, progress_interval, logger)


def process_zst_file_multi(input_file: str, line_processor: Callable[[str, Dict], Dict[str, Any]],
                          processor_state: Dict[str, Any], progress_interval: int = 10_000_000, logger=None) -> Dict[str, int]:
    """
    Process a compressed file line by line with multi-output support.

    Args:
        input_file: Path to input .zst file
        line_processor: Function that takes (line, state) and returns:
                       {'matched': bool, 'output_files': List[str], 'data': Any, 'state_updates': Dict}
                       or {'matched': False} for skipped lines
        processor_state: Mutable state dict passed to line_processor
        progress_interval: Log progress every N lines
        logger: Optional logger for progress messages (if None, uses print)

    Returns:
        Dictionary with processing statistics including per-output stats
    """
    stats = {
        "lines_processed": 0,
        "lines_matched": 0,
        "error_lines": 0,
        "output_stats": {}
    }
    start_time = time.time()

    file_size = os.path.getsize(input_file)
    msg = f"Processing {os.path.basename(input_file)} ({file_size / (1024**3):.1f} GB)"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    # Track open writers for multiple outputs
    open_writers = {}

    try:
        # Open input file
        with open(input_file, 'rb') as f:
            reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(f)

            buffer = ''
            while True:
                chunk = read_and_decode(reader)
                if not chunk:
                    break

                lines = (buffer + chunk).split("\n")

                for line in lines[:-1]:
                    if line.strip():
                        stats["lines_processed"] += 1

                        # Progress logging
                        if stats["lines_processed"] % progress_interval == 0:
                            elapsed = time.time() - start_time
                            rate = stats["lines_processed"] / elapsed if elapsed > 0 else 0
                            progress_msg = (f"  Progress: {stats['lines_processed']:,} lines, "
                                          f"{stats['lines_matched']:,} matched ({rate:,.0f} lines/sec)")
                            if logger:
                                logger.info(progress_msg)
                            else:
                                print(progress_msg)

                        try:
                            result = line_processor(line.strip(), processor_state)

                            if result.get('matched', False):
                                output_files = result.get('output_files', [])
                                data = result.get('data', line.strip())

                                # Write to each specified output file
                                for output_file in output_files:
                                    # Lazy-open writers
                                    if output_file not in open_writers:
                                        ensure_directory(output_file)
                                        file_handle = open(output_file, 'wb')
                                        open_writers[output_file] = zstandard.ZstdCompressor(level=3, threads=4).stream_writer(file_handle)
                                        stats["output_stats"][output_file] = 0

                                    # Write data
                                    writer = open_writers[output_file]
                                    if isinstance(data, str):
                                        writer.write(data.encode('utf-8'))
                                    else:
                                        writer.write(json_dumps(data).encode('utf-8'))
                                    writer.write(b'\n')

                                    stats["output_stats"][output_file] += 1

                                # Update processor state if provided
                                state_updates = result.get('state_updates', {})
                                processor_state.update(state_updates)

                                stats["lines_matched"] += 1

                        except Exception:
                            stats["error_lines"] += 1

                buffer = lines[-1]

            reader.close()

        # Close all writers
        for output_file, writer in open_writers.items():
            writer.close()

    except Exception as e:
        # Ensure all writers are closed on error
        for writer in open_writers.values():
            try:
                writer.close()
            except:
                pass
        msg = f"Error processing {input_file}: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        raise

    elapsed = time.time() - start_time
    rate = stats["lines_processed"] / elapsed if elapsed > 0 else 0
    msg = (f"Completed {os.path.basename(input_file)}: {stats['lines_processed']:,} lines, "
           f"{stats['lines_matched']:,} matched in {elapsed:.1f}s ({rate:,.0f} lines/sec)")
    if logger:
        logger.info(msg)
    else:
        print(msg)

    return stats


def read_zst_lines(file_path: str, max_lines: int = None) -> List[str]:
    """
    Read lines from a compressed file.

    Args:
        file_path: Path to .zst file
        max_lines: Maximum number of lines to read (None = all)

    Returns:
        List of lines (strings)
    """
    lines = []
    lines_read = 0

    with open(file_path, 'rb') as f:
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(f)

        buffer = ''
        while True:
            chunk = read_and_decode(reader)
            if not chunk:
                break

            chunk_lines = (buffer + chunk).split("\n")

            for line in chunk_lines[:-1]:
                if line.strip():
                    lines.append(line.strip())
                    lines_read += 1

                    if max_lines and lines_read >= max_lines:
                        reader.close()
                        return lines

            buffer = chunk_lines[-1]

        reader.close()

    return lines


def write_json_file(data: Any, file_path: str, pretty: bool = False):
    """Write data to JSON file.

    Args:
        data: Data to write
        file_path: Path to write to
        pretty: If True, use indentation for readability (for stats/summary files)
                If False, use compact format (for data files like comments/submissions)
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        if pretty:
            f.write(json_dumps_pretty(data))
        else:
            f.write(json_dumps(data))


def read_json_file(file_path: str) -> Any:
    """Read data from JSON file."""
    with open(file_path, 'r') as f:
        return json.loads(f.read())


def get_files_in_date_range(folder: str, prefix: str, date_range: Tuple[str, str], logger=None) -> List[str]:
    """
    Get files in a folder that match prefix and fall within date range.

    Args:
        folder: Directory to search
        prefix: File prefix (e.g., "RC_", "RS_")
        date_range: Tuple of (start_date, end_date) in YYYY-MM format
        logger: Optional logger for messages (if None, uses print)

    Returns:
        List of file paths sorted by date (newest first)
    """
    if not os.path.exists(folder):
        msg = f"Warning: Directory {folder} does not exist"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return []

    start_date, end_date = date_range
    files = []

    for filename in os.listdir(folder):
        if filename.startswith(prefix) and filename.endswith('.zst') and not filename.endswith('corrupted.zst'):
            try:
                # Extract date from filename (e.g., RC_2023-01.zst -> 2023-01)
                date_part = filename.split('_')[1].split('.')[0]

                if start_date <= date_part <= end_date:
                    files.append(os.path.join(folder, filename))

            except (IndexError, ValueError):
                msg = f"Warning: Could not parse date from filename: {filename}"
                if logger:
                    logger.warning(msg)
                else:
                    print(msg)

    # Sort by date (newest first)
    files.sort(reverse=True)
    msg = f"Found {len(files)} {prefix} files in date range {start_date} to {end_date}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    return files


def process_files_parallel(files: List[str], process_func: Callable, processes: int = None, logger=None) -> List[Any]:
    """
    Process multiple files in parallel.

    Args:
        files: List of file paths or argument tuples
        process_func: Function to process each file
        processes: Number of parallel processes (default: from config)
        logger: Optional logger for messages (if None, uses print)

    Returns:
        List of results from processing
    """
    if processes is None:
        from config import PROCESSES
        processes = PROCESSES

    msg = f"Processing {len(files)} files with {processes} processes"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    start_time = time.time()

    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.map(process_func, files)

    elapsed = time.time() - start_time
    msg = f"Parallel processing completed in {elapsed:.1f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    return results


def get_file_size_gb(file_path: str) -> float:
    """Get file size in GB."""
    return os.path.getsize(file_path) / (1024**3)


def write_zst_lines(file_path: str, lines: List[str], level: int = 3, threads: int = 4):
    """
    Write lines to a compressed file.

    Args:
        file_path: Path to output .zst file
        lines: List of strings to write (one per line)
        level: Compression level (1-22, default 3)
        threads: Number of threads for compression (default 4)
    """
    ensure_directory(file_path)

    with open(file_path, 'wb') as f:
        compressor = zstandard.ZstdCompressor(level=level, threads=threads)
        with compressor.stream_writer(f) as writer:
            for line in lines:
                writer.write(line.encode('utf-8'))
                writer.write(b'\n')


def write_zst_json_objects(file_path: str, objects: List[Any], level: int = 3, threads: int = 4):
    """
    Write JSON objects to a compressed file.

    Args:
        file_path: Path to output .zst file
        objects: List of objects to write as JSON (one per line)
        level: Compression level (1-22, default 3)
        threads: Number of threads for compression (default 4)
    """
    ensure_directory(file_path)

    with open(file_path, 'wb') as f:
        compressor = zstandard.ZstdCompressor(level=level, threads=threads)
        with compressor.stream_writer(f) as writer:
            for obj in objects:
                json_line = json_dumps(obj)
                writer.write(json_line.encode('utf-8'))
                writer.write(b'\n')


def ensure_directory(file_path: str):
    """Ensure directory exists for a file path."""
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)