import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, FloatType, ArrayType, DoubleType,IntegerType
import timeit
from pyspark import SparkConf, SparkContext
class Project3:
    def run(self, inputpath, outputpath, k):
        tau = float(k)
        
        conf = SparkConf().setAppName("TransactionSimilarity")
        sc = SparkContext(conf=conf)
        
        spark = SparkSession(sc)

        rdd = sc.textFile(inputpath)

        schema = StructType([
            StructField("TransactionID", IntegerType(), True),
            StructField("ProductName", StringType(), True),
            StructField("TransactionDate", StringType(), True),  
            StructField("Amount", FloatType(), True)     
        ])
        
        def parse_line(row):
            cols = row.split(",")
            if len(cols) < 4:
                print(f"Skipping malformed row: {row}")
                return None  
            try:
                return (
                    int(cols[0]),
                    cols[1],
                    cols[2][3:9],  
                    float(cols[3])
                )
            except ValueError:
                print(f"Skipping row with invalid float conversion: {row}")
                return None  
        
        parsed_rdd = rdd.map(parse_line).filter(lambda x: x is not None)
        df = parsed_rdd.toDF(schema=schema)
        df = df.dropDuplicates(["TransactionID", "ProductName"])
        start_time = timeit.default_timer()

        # 计算每个ProductName的频率
        product_counts = df.rdd.map(lambda row: (row['ProductName'], 1)) \
                               .reduceByKey(lambda a, b: a + b)

        # 按照频率从小到大排序并转换为DataFrame
        sorted_counts = product_counts.sortBy(lambda x: x[1])
        product_counts_df = sorted_counts.toDF(["ProductName", "TotalCount"])

        # 将原始数据与频率数据进行连接并按频率排序
        sorted_df = df.rdd.map(lambda row: (row['ProductName'], (row['TransactionID'], row['TransactionDate']))) \
                          .join(product_counts) \
                          .map(lambda x: (x[1][0][0], x[1][0][1], x[0], x[1][1])) \
                          .toDF(["TransactionID", "TransactionDate", "ProductName", "TotalCount"])

        # 按TransactionID分组并收集ProductName
        grouped_df = sorted_df.groupBy("TransactionID", "TransactionDate") \
                              .agg(collect_list("ProductName").alias("ProductNamesSorted"))

        def get_prefix_length(length, tau):
            return int((1 - tau) * length) + 1

        prefix_udf = udf(lambda products: products[:get_prefix_length(len(products), tau)], ArrayType(StringType()))

        final_df = grouped_df.withColumn("Prefix", prefix_udf(col("ProductNamesSorted")))
        

        def group_tokens(token):
            if token[0] in 'ABCDEF':
                return 'W'
            elif token[0] in 'GHIJKL':
                return 'X'
            elif token[0] in 'MNOPQR':
                return 'Y'
            else:
                return 'Z'

        group_tokens_udf = udf(group_tokens, StringType())
        
        prefixes = final_df.select("TransactionID", "TransactionDate", explode(col("Prefix")).alias("Prefix"))
        prefixes = prefixes.withColumn("GroupedPrefix", group_tokens_udf(col("Prefix")))
        

        joined_df = prefixes.alias("df1").join(prefixes.alias("df2"), "GroupedPrefix") \
            .where(((col("df1.TransactionID")) < col("df2.TransactionID")) & (col("df1.TransactionDate") != col("df2.TransactionDate")))

        joined_df = joined_df.select("GroupedPrefix","df1.TransactionID", "df2.TransactionID","df1.Prefix","df2.Prefix")
        
        joined_df = joined_df.groupBy("df1.TransactionID", "df2.TransactionID") \
            .agg(collect_set("GroupedPrefix").alias("CommonPrefix"))
        
        def jaccard_similarity(products1, products2):
            set1, set2 = set(products1), set(products2)
            return float(len(set1 & set2)) / float(len(set1 | set2))

        jaccard_udf = udf(lambda products1, products2: jaccard_similarity(products1, products2), DoubleType())

        similarity_df = joined_df \
            .join(final_df.alias("f1"), (col("df1.TransactionID") == col("f1.TransactionID")) ) \
            .join(final_df.alias("f2"), (col("df2.TransactionID") == col("f2.TransactionID")) ) \
            .withColumn("Similarity", jaccard_udf(col("f1.ProductNamesSorted"), col("f2.ProductNamesSorted"))) \
            .filter(col("Similarity") >= tau) \
            .select(col("df1.TransactionID").alias("TransactionID1"), col("df2.TransactionID").alias("TransactionID2"), "Similarity")
        
        similarity_df = similarity_df.orderBy("TransactionID1")
        
        formatted_df = similarity_df.withColumn(
            "FormattedResult",
            concat(
                lit("("), col("TransactionID1"), lit(","), col("TransactionID2"), lit("):"), col("Similarity")
            )
        )

        formatted_df = formatted_df.select("FormattedResult")
        
        formatted_df.coalesce(1).write.text(outputpath)
        
        end_time = timeit.default_timer()
        print(f"Execution time: {end_time - start_time} seconds")

        spark.stop()

if __name__ == '__main__':
    Project3().run(sys.argv[1], sys.argv[2], sys.argv[3])
