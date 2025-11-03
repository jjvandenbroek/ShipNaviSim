import glob
import polars as pl
from datetime import timedelta

# Columns: SHIP_ID;MMSI;IMO;LAT;LON;TIMESTAMP_UTC;STATUS;HEADING;COURSE;SPEED_KNOTSX10
data_path = "./raw_data"
files = glob.glob(f"{data_path}/*Positions*.csv")
files = sorted(files)

SHIP_ID_COL = "SHIP_ID"
TIME_COL = "TIMESTAMP_UTC"

def seperate_chunks(df, time_theshold, minimum_sample_count, min_duration_seconds=None):
    """
    Separate continuous vessel trajectories into chunks based on time gaps.
    
    Args:
        df: Polars DataFrame with vessel position data
        time_theshold: Maximum time gap between consecutive points in same chunk
        minimum_sample_count: Minimum number of AIS records required for a chunk
        min_duration_seconds: Minimum time span (seconds) for a chunk to be valid.
                             If None, defaults to 2x the interpolation interval (20 seconds)
    
    Returns:
        Polars DataFrame with chunks, filtered by sample count and duration
    """
    if min_duration_seconds is None:
        # Default: require at least 20 seconds of data (2x interpolation interval)
        # This ensures enough data for proper interpolation
        min_duration_seconds = 20
    
    df = df.lazy()
    df = df.sort([SHIP_ID_COL, TIME_COL])

    # Create a new column for the time difference
    df = df.with_columns([
        pl.col(TIME_COL).diff().dt.cast_time_unit('ms').alias('time_diff'),
        pl.col(SHIP_ID_COL).diff().alias('ship_id_diff')
    ])
    
    # Create a chunk_id column
    df = df.with_columns([
        pl.when((pl.col('time_diff') > time_theshold) | (pl.col('ship_id_diff') != 0))
        .then(1)
        .otherwise(0)
        .cum_sum()
        .alias('chunk_id')
    ])
        
    # Group by chunk_id and aggregate
    result = df.group_by('chunk_id').agg([
        pl.col(SHIP_ID_COL).first().alias(SHIP_ID_COL),
        pl.col(TIME_COL).min().alias('start_time'),
        pl.col(TIME_COL).max().alias('end_time'),
        pl.count(TIME_COL).alias('num_records'),
        pl.struct(pl.all()).alias('all_records')
    ]).sort([SHIP_ID_COL, 'start_time'])
    
    # Add duration column (in seconds)
    result = result.with_columns([
        ((pl.col('end_time') - pl.col('start_time')).dt.total_seconds()).alias('duration_seconds')
    ])
    
    # Filter by both record count AND duration
    # This ensures chunks have enough data points AND enough time span for interpolation
    result = result.filter(
        (pl.col('num_records') >= minimum_sample_count) & 
        (pl.col('duration_seconds') >= min_duration_seconds)
    ).collect()
    
    return result
