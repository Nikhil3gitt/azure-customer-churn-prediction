"""Sessionize clickstream and create engagement features using PySpark."""

from pyspark.sql import SparkSession
from pyspark.sql.functions import unix_timestamp, lag, sum as _sum
from pyspark.sql.window import Window

def build_features(input_path: str, output_path: str):
    spark = SparkSession.builder.appName("SessionizeClickstream").getOrCreate()
    df = spark.read.parquet(input_path)

    # Define session gaps (30m)
    w = Window.partitionBy('customer_id').orderBy('event_time')
    df = df.withColumn('prev_ts', lag('event_time').over(w))
    df = df.withColumn('gap', unix_timestamp('event_time') - unix_timestamp('prev_ts'))
    df = df.withColumn('new_session', (df.gap > 1800) | df.prev_ts.isNull())
    df = df.withColumn('session_id', _sum(df.new_session.cast('int')).over(w))

    # Example engagement metrics
    features = (
        df.groupBy('customer_id', 'session_id')
          .agg(
              _sum('page_view').alias('pages'),
              _sum('cart_add').alias('cart_adds')
          )
    )
    features.write.mode('overwrite').parquet(output_path)
    spark.stop()

if __name__ == "__main__":
    import sys
    build_features(sys.argv[1], sys.argv[2])
