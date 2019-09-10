using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using MovieRecommender.Models;
using Newtonsoft.Json;
using System;
using System.IO;
using System.Threading.Tasks;

namespace MovieRecommender
{
    public static class MovieRecommender
    {
        [FunctionName("MovieRecommender")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get", Route = null)] HttpRequest req,
            ILogger log)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            string movieData = req.Query["movieData"];

            if (String.IsNullOrWhiteSpace(movieData))
            {
                return new BadRequestObjectResult("Please pass a name on the query string");
            }

            // Create test input & make single prediction
            String[] values = movieData.Split(":");
            var movieRatingTestInput = new MovieRating
            {
                userId = Int32.Parse(values[0]),
                movieId = Int32.Parse(values[1])
            };

            // Create MLContext to be shared across the model creation workflow objects 
            MLContext mlContext = new MLContext();

            // Load data
            (IDataView trainingDataView, IDataView testDataView) = LoadData(mlContext);

            // Build & train model
            ITransformer model = BuildAndTrainModel(mlContext, trainingDataView);

            // Evaluate quality of model
            EvaluateModel(mlContext, testDataView, model);

            // Use model to try a single prediction (one row of data)
            string result = UseModelForSinglePrediction(mlContext, model, movieRatingTestInput);

            // Save model
            SaveModel(mlContext, trainingDataView.Schema, model);

            return (ActionResult)new OkObjectResult(result);
        }

        // Load data
        public static (IDataView training, IDataView test) LoadData(MLContext mlContext)
        {
            // Load training & test datasets using datapaths
            var trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-train.csv");
            var testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-test.csv");

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
            IDataView testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader: true, separatorChar: ',');

            return (trainingDataView, testDataView);
        }

        // Build and train model
        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
        {
            // Add data transformations
            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"));

            // Set algorithm options and append algorithm
            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "movieIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

            Console.WriteLine("=============== Training the model ===============");
            ITransformer model = trainerEstimator.Fit(trainingDataView);

            return model;
        }

        // Evaluate model
        public static void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
        {
            // Evaluate model on test data & print evaluation metrics
            Console.WriteLine("=============== Evaluating the model ===============");
            var prediction = model.Transform(testDataView);

            var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
            Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
        }

        // Use model for single prediction
        public static String UseModelForSinglePrediction(MLContext mlContext, ITransformer model, MovieRating movieRatingTestInput)
        {
            String result = String.Empty;
            Console.WriteLine("=============== Making a prediction ===============");
            var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

            var movieRatingPrediction = predictionEngine.Predict(movieRatingTestInput);

            if (Math.Round(movieRatingPrediction.Score, 1) > 3.5)
            {
                result = "Movie " + movieRatingTestInput.movieId + " is recommended for user " + movieRatingTestInput.userId;
            }
            else
            {
                result = "Movie " + movieRatingTestInput.movieId + " is not recommended for user " + movieRatingTestInput.userId;
            }

            Console.WriteLine(result);
            return result;
        }

        public static void SaveModel(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            // Save the trained model to .zip file
            var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "MovieRecommenderModel.zip");

            Console.WriteLine("=============== Saving the model to a file ===============");
            mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
        }
    }
}