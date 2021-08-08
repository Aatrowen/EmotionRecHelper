using System.Collections.Generic;
using EmotionRecHelper.DataStructures;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace EmotionRecHelper
{
    class OnnxModelScorer
    {
        private readonly string _modelLocation;
        private readonly MLContext _mlContext;

        public OnnxModelScorer(string modelLocation, MLContext mlContext)
        {
            _modelLocation = modelLocation;
            _mlContext = mlContext;
        }

        /// <summary>
        /// The size of the input(image) to the model
        /// </summary>
        private struct ImageNetSettings
        {
            public const int ImageHeight = 64;
            public const int ImageWidth = 64;
        }

        /// <summary>
        /// The name of the input-output tensor of the model
        /// </summary>
        private struct ErModelSettings
        {
            public const string ModelInput = "Input3";

            public const string ModelOutput = "Plus692_Output_0";
        }

        private ITransformer LoadModel(string modelLocation)
        {
            // Create IDataView from empty list to obtain input data schema
            var data = _mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());

            // Define scoring pipeline
            var pipeline = _mlContext.Transforms.LoadImages(outputColumnName: "Input3", imageFolder: "", inputColumnName: nameof(ImageNetData.ImagePath))
                .Append(_mlContext.Transforms.ResizeImages(outputColumnName: "Input3", imageWidth: ImageNetSettings.ImageWidth, imageHeight: ImageNetSettings.ImageHeight, inputColumnName: "Input3"))
                .Append(_mlContext.Transforms.ExtractPixels(outputColumnName: "Input3", colorsToExtract: ImagePixelExtractingEstimator.ColorBits.Blue))
                .Append(_mlContext.Transforms.ApplyOnnxModel(modelFile: modelLocation, outputColumnNames: new[] { ErModelSettings.ModelOutput }, inputColumnNames: new[] { ErModelSettings.ModelInput }));

            // Fit scoring pipeline
            var model = pipeline.Fit(data);

            return model;
        }

        private static IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model)
        {
            var scoredData = model.Transform(testData);

            var probabilities = scoredData.GetColumn<float[]>(ErModelSettings.ModelOutput);

            return probabilities;
        }

        public IEnumerable<float[]> Score(IDataView data)
        {
            var model = LoadModel(_modelLocation);

            return PredictDataUsingModel(data, model);
        }
    }
}
