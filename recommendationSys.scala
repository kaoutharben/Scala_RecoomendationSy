import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import scala.Tuple2
import org.apache.spark.rdd.RDD

val ratigsFile = "data/ratings.csv"
val df1 = spark.read.format("com.databricks.spark.csv").option("header", true).load(ratigsFile)
val ratingsDF = df1.select(df1.col("userId"), df1.col("movieId"), df1.col("rating"), df1.col("timestamp"))ratingsDF.show(false)

val moviesFile = "data/movies.csv"
val df2 = spark.read.format("com.databricks.spark.csv").option("header", "true").load(moviesFile)
val moviesDF = df2.select(df2.col("movieId"), df2.col("title"), df2.col("genres"))

ratingsDF.createOrReplaceTempView("ratings")
moviesDF.createOrReplaceTempView("movies")

val results = spark.sql("select movies.title, movierates.maxr, movierates.minr, movierates.cntu "
+ "from(SELECT ratings.movieId,max(ratings.rating) as maxr,"
+ "min(ratings.rating) as minr,count(distinct userId) as cntu "
+ "FROM ratings group by ratings.movieId) movierates "
+ "join movies on movierates.movieId=movies.movieId "
+ "order by movierates.cntu desc") results.show(false)

val results2 = spark.sql(
"SELECT ratings.userId, ratings.movieId,"
+ "ratings.rating, movies.title FROM ratings JOIN movies"
+ "ON movies.movieId=ratings.movieId"
+ "where ratings.userId=668 and ratings.rating > 4") results2.show(false)

val splits = ratingsDF.randomSplit(Array(0.75, 0.25), seed = 12345L)
val (trainingData, testData) = (splits(0), splits(1))
val numTraining = trainingData.count()
val numTest = testData.count()
println("Training: " + numTraining + " test: " + numTest)

val ratingsRDD = trainingData.rdd.map(row => {
val userId = row.getString(0)
val movieId = row.getString(1)
val ratings = row.getString(2)
Rating(userId.toInt, movieId.toInt, ratings.toDouble)
})

val testRDD = testData.rdd.map(row => {
val userId = row.getString(0)
val movieId = row.getString(1)
val ratings = row.getString(2)
Rating(userId.toInt, movieId.toInt, ratings.toDouble)
})

val rank = 20
val numIterations = 15
val lambda = 0.10
val alpha = 1.00 val block = -1
val seed = 12345L
val implicitPrefs = false
val model = new ALS().setIterations(numIterations) .setBlocks(block).setAlpha(alpha)
.setLambda(lambda)
.setRank(rank) .setSeed(seed)
.setImplicitPrefs(implicitPrefs)
.run(ratingsRDD)

println("Rating:(UserID, MovieID, Rating)")
println("----------------------------------")
val topRecsForUser = model.recommendProducts(668, 6) for (rating <- topRecsForUser) { println(rating.toString()) } println("----------------------------------")
