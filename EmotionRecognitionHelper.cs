using System;
using System.Collections.Generic;
using System.Linq;
using EmotionRecHelper.DataStructures;
using Microsoft.ML;

namespace EmotionRecHelper
{
    public class EmotionRecognitionHelper : IEmotionRecHelper
    {
        private string modelFilePath = @"Model/EmotionRecognition.onnx";
        private MLContext MlContext { get; set; }
        private OnnxModelScorer ModelScorer { get; set; }

        public EmotionRecognitionHelper()
        {
            MlContext = new MLContext();

            // Create instance of model scorer
            ModelScorer = new OnnxModelScorer(modelFilePath, MlContext);
        }

        public Dictionary<string, string> GetEmotionFromDirectory(string imagesFolder)
        {
            var res = new Dictionary<string, string>();
            try
            {
                // Load Data
                var images = ImageNetData.ReadFromDirectory(imagesFolder);
                var imageDataView = MlContext.Data.LoadFromEnumerable(images);

                // Use model to score data
                var probabilities = ModelScorer.Score(imageDataView);
                var resPath = images.Select(x => x.ImagePath).ToList();
                var index = 0;
                foreach (var probable in probabilities)
                {
                    var scores = Softmax(probable);

                    var (topResultIndex, topResultScore) = scores.Select((predictedClass, index) => (Index: index, Value: predictedClass))
                        .OrderByDescending(result => result.Value)
                        .First();

                    res.Add(resPath[index], Labels[topResultIndex]);
                    index ++;
                }

                return res;
            }
            catch (Exception ex)
            {
                res.Add("Error", ex.ToString());
                return res;
            }
        }

        public string GetEmotionFromAbsoluteFileName(string imageName)
        {
            try
            {
                // Load Data
                var image = ImageNetData.ReadFromFile(imageName);
                var imageDataView = MlContext.Data.LoadFromEnumerable(image);

                // Use model to score data
                var probabilities = ModelScorer.Score(imageDataView);

                string res = string.Empty;
                foreach (var probable in probabilities)
                {
                    var scores = Softmax(probable);

                    var (topResultIndex, topResultScore) = scores.Select((predictedClass, index) => (Index: index, Value: predictedClass))
                        .OrderByDescending(result => result.Value)
                        .First();

                    res = Labels[topResultIndex];
                }
                return res;
            }
            catch (Exception ex)
            {
                return ex.ToString();
            }
        }

        private static float[] Softmax(float[] values)
        {
            var maxVal = values.Max();
            var exp = values.Select(v => Math.Exp(v - maxVal));
            var sumExp = exp.Sum();

            return exp.Select(v => (float)(v / sumExp)).ToArray();
        }

        private static readonly string[] Labels = {
            "neutral",
            "happiness",
            "surprise",
            "sadness",
            "anger",
            "disgust",
            "fear",
            "contempt"
        };
    }
}
