using System;
using System.Linq;

namespace CommwellEcgCalculation
{
    class Morph
    {
        public int size = 200;
        public int[] wave { get; set; }
        public int numberOfWaves { get; set; }

        public Morph(int size1)
        {
            size = size1;
            wave = new int[size];
            numberOfWaves = 0;

        }
        public void addWave(int[] data)
        {
            if (data.Length < size) return;  //should not happen error
            for (int i = 0; i < size; i++) wave[i] += data[i];
            numberOfWaves++;
        }

        public double compareWave(int[] data)
        {
            int[] wave1 = new int[size];
            wave1 = averageWave();
            double avg1 = (double)wave1.Average();
            double avg2 = (double)data.Average();
            var sum1 = wave1.Zip(data, (x1, y1) => (x1 - avg1) * (y1 - avg2)).Sum();
            var sumSqr1 = wave1.Sum(x => Math.Pow((x - avg1), 2.0));
            var sumSqr2 = data.Sum(y => Math.Pow((y - avg2), 2.0));

            var result = sum1 / Math.Sqrt(sumSqr1 * sumSqr2);

            return result; // Math.Abs( result);
        }

        public double compareWaveHeight(int[] data)
        {
            int[] wave1 = new int[size];
            wave1 = averageWave();
            int height1 = 0;
            int height2 = 0;
            for (int i = 120; i < 180; i++)
            {
                height1 += wave1[i];
                height2 += data[i];
            }

            return Math.Abs(1.0 - (double)(height1 - height2) / height1);
        }
        public bool validateMorph()
        {
            bool ret = true;
            int[] wave1 = new int[size];
            int max = 0;
            int maxIndex = 0;
            wave1 = averageWave();
            for (int i = 0; i < wave1.Length; i++)
                if (Math.Abs(wave1[i]) > max)
                {
                    max = Math.Abs(wave1[i]);
                    maxIndex = i;
                }
            if ((maxIndex < wave1.Length / 2 - 20) || (maxIndex > wave1.Length / 2 + 20))
            {
                clearMorph();
                ret = false;
            }
            return ret;
        }
        public void clearMorph()
        {
            for (int i = 0; i < wave.Length; i++) wave[i] = 0;
            numberOfWaves = 0;
        }

        public int[] averageWave()
        {
            int[] avg = new int[size];
            for (int i = 0; i < size; i++)
            {
                if (numberOfWaves > 0) avg[i] = wave[i] / numberOfWaves;
                else avg[i] = wave[i];
            }
            return avg;
        }
    }


}
