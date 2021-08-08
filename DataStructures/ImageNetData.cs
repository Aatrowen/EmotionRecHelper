using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;

namespace EmotionRecHelper.DataStructures
{
    public class ImageNetData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)] 
        private string Label;

        /// <summary>
        /// Get the specified picture information
        /// </summary>
        /// <param name="imageName"></param>
        /// <returns></returns>
        public static IEnumerable<ImageNetData> ReadFromFile(string imageName)
        {
            FileInfo image = new(imageName);
            List<FileInfo> res = new() { image};
            return res
                .Select(fileInfo => new ImageNetData { ImagePath = fileInfo.FullName, Label = fileInfo.Name });
        }

        /// <summary>
        /// Get all the pictures from imageFolder directory
        /// </summary>
        /// <param name="imageFolder"></param>
        /// <returns></returns>
        public static IEnumerable<ImageNetData> ReadFromDirectory(string imageFolder)
        {
            List<FileInfo> imageFiles = new();
            DirectoryInfo currentDir = new(imageFolder);
            foreach (var directoryInfo in currentDir.GetDirectories())
            {
                imageFiles.AddRange(directoryInfo.GetFiles("*.png"));
                imageFiles.AddRange(directoryInfo.GetFiles("*.jpg"));
            }
            imageFiles.AddRange(currentDir.GetFiles("*.png"));
            imageFiles.AddRange(currentDir.GetFiles("*.jpg"));

            return imageFiles
                .Select(fileInfo => new ImageNetData { ImagePath = fileInfo.FullName, Label = fileInfo.Name });
        }
    }
}
