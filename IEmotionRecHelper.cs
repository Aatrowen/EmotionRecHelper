using System.Collections.Generic;

namespace EmotionRecHelper
{
    public interface IEmotionRecHelper
    {
        /// <summary>
        /// Get pictures from the specified directory and return the results of emotion
        /// </summary>
        /// <param name="imagesFolder"></param>
        /// <returns></returns>
        public Dictionary<string, string> GetEmotionFromDirectory(string imagesFolder);

        /// <summary>
        /// Returns its emotion recognition result according to the specified absolute path of the picture
        /// </summary>
        /// <param name="imageName"></param>
        /// <returns></returns>
        public string GetEmotionFromAbsoluteFileName(string imageName);
    }
}
