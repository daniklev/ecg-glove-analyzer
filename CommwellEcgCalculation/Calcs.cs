using System;


namespace CommwellEcgCalculation
{
    class Calcs
    {
        int[] ECGarray;
        int[] dArray;
        int[] d2Array;
        int[] data1;
        int[] data2;
        int[] data3;
        int qPoint, pPoint, sPoint, tPoint;
        int row_number = 0;
        bool first = true;
        int col = 0;
        int rr;
        enum rowNums
        {
            pStart, pPeak, pVal, qStart, qPeak, qVal, rPeak, rVal, sPeak, sVal, tPeak, tVal, tEnd, prTime, qrsTime, qtTime,
            qtcTime, pAxis, qrsAxis, tAxis, hr, rr, end
        };
        string[] rowNumsLabels = new string[]
        {
            "pStart", "pPeak", "pVal", "qStart", "qPeak", "qVal", "rPeak", "rVal",
            "sPeak", "sVal", "tPeak", "tVal", "tEnd", "prTime", "qrsTime", "qtTime",
            "qtcTime", "pAxis", "qrsAxis", "tAxis", "hr", "rr", "end"
        };

        public enum val_labels
        { p, pr, qrs, qt, qtcb, paxis, qrsaxis, taxis, rr, end };

        public int[] val = new int[(int)val_labels.end];


        int[][] data = new int[4][];
        public Calcs(int[] _data1, int[] _data2, int[] _data3, int _rr)
        {
            data1 = _data1;
            data2 = _data2;
            data3 = _data3;
            rr = _rr;
            ECGarray = new int[data1.Length];
            dArray = new int[data1.Length - 1];
            d2Array = new int[data1.Length - 2];
            for (int i = 0; i < 4; i++)
                data[i] = new int[(int)rowNums.end];
            setData(data1, 0);
            MakeCalcs();
            setData(data2, 1);
            MakeCalcs1();
            setData(data3, 2);
            MakeCalcs1();
            double r1Total = (int)data[0][(int)rowNums.qPeak] +
              (int)data[0][(int)rowNums.rPeak] +
              (int)data[0][(int)rowNums.sPeak];

            double r2Total = (int)data[2][(int)rowNums.qPeak] +
              (int)data[2][(int)rowNums.rPeak] +
              (int)data[2][(int)rowNums.sPeak];

            double hyp = Math.Sqrt(r1Total * r1Total + r2Total * r2Total);
            double angle = Math.Asin(r2Total / hyp) * 180.0 / Math.PI;
            val[(int)val_labels.qrsaxis] = (int)angle;

            //compute P Axis
            r1Total = (int)data[0][(int)rowNums.pPeak];
            r2Total = (int)data[2][(int)rowNums.pPeak];
            hyp = Math.Sqrt(r1Total * r1Total + r2Total * r2Total);
            angle = Math.Asin(r2Total / hyp) * 180.0 / Math.PI;
            val[(int)val_labels.paxis] = (int)angle;

            //compute T Axis
            r1Total = (int)data[0][(int)rowNums.tPeak];
            r2Total = (int)data[2][(int)rowNums.tPeak];
            hyp = Math.Sqrt(r1Total * r1Total + r2Total * r2Total);
            angle = Math.Asin(r2Total / hyp) * 180.0 / Math.PI;
            val[(int)val_labels.taxis] = (int)angle;
            val[(int)val_labels.rr] = rr;

        }
        private void setData(int[] data, int colNum)
        {
            ECGarray = data;

            for (int i = 1; i < dArray.Length; i++)
                dArray[i] = ECGarray[i] - ECGarray[i - 1];
            for (int i = 1; i < d2Array.Length; i++)
                d2Array[i] = dArray[i] - dArray[i - 1];
            col = colNum;

        }
        private void MakeCalcs1()
        {
            try
            {

                int max = -10000;
                int min = 10000;
                int xOffset = 0;
                int rMax1 = 0;
                int sTime = 0;
                int qTime = 0;
                int qrsTime = 0;
                int pMax = 0;
                int tMax = 0;
                int length = ECGarray.Length;
                int x;
                for (int i = 0; i < length; i++)
                {
                    if (ECGarray[i] > max) max = ECGarray[i];
                    if (ECGarray[i] < min) min = ECGarray[i];
                }

                rMax1 = data[0][(int)rowNums.rVal];
                data[col][(int)rowNums.rPeak] = (FindMinMax(rMax1));

                qTime = data[0][(int)rowNums.qVal];
                data[col][(int)rowNums.qPeak] = FindMinMax(qTime);

                sTime = data[0][(int)rowNums.sVal];
                data[col][(int)rowNums.sPeak] = FindMinMax(sTime);

                qrsTime = 2 * (sTime - qTime);
                data[col][(int)rowNums.qrsTime] = qrsTime;

                pMax = data[0][(int)rowNums.pVal];
                data[col][(int)rowNums.pVal] = pMax;
                data[col][(int)rowNums.pPeak] = FindMinMax(pMax);

                tMax = data[0][(int)rowNums.tVal];
                data[col][(int)rowNums.tVal] = tMax;
                data[col][(int)rowNums.tPeak] = FindMinMax(tMax);


            }
            catch (Exception e)
            {
                return;
            }

        }

        public void MakeCalcs()
        {
            int max = -10000;
            int min = 10000;
            int xOffset = 0;
            int rMax1 = 0;
            int sTime = 0;
            int qTime = 0;
            int qrsTime = 0;
            int pMax = 0;
            int length = ECGarray.Length;
            int x;
            for (int i = 0; i < length; i++)
            {
                if (ECGarray[i] > max) max = ECGarray[i];
                if (ECGarray[i] < min) min = ECGarray[i];
            }


            rMax1 = FindRmax(ECGarray);
            data[0][(int)rowNums.rVal] = rMax1;
            qTime = FindQ(rMax1);
            //data[0][(int)rowNums.qPeak] = qTime;
            sTime = FindS(rMax1);
            qrsTime = 2 * (sTime - qTime);
            data[0][(int)rowNums.qrsTime] = qrsTime;
            pMax = FindPmax(qTime);
            int pTime = FindP(pMax);
            x = pTime + xOffset;
            int pLength = FindPlength(pMax) * 2;
            val[(int)val_labels.p] = pLength;
            int prTime = 2 * (qTime - pTime);
            val[(int)val_labels.p] = pLength;
            val[(int)val_labels.pr] = prTime;
            data[col][(int)rowNums.prTime] = prTime;
            int tMax = FindTmax(sTime, qTime);
            int tTime = FindT(tMax, qTime);
            val[(int)val_labels.qrs] = ((sTime - qTime) * 2);
            x = tTime + xOffset;

            int qtTime = 2 * (tTime - qTime);
            val[(int)val_labels.qt] = qtTime;
            double qtcb = (double)qtTime / (Math.Sqrt((double)rr / 1000));
            val[(int)val_labels.qtcb] = (int)qtcb;
        }
        private int GetRmax(int[] data, int start)
        {
            int ret = 0;
            int max = 0;
            if (((start - 20) > 0) && ((start + 20) < data.Length))
            {
                for (int i = start - 20; i < start + 20; i++)
                {
                    if (Math.Abs(data[i]) > max)
                    {
                        ret = i;
                        max = Math.Abs(data[i]);
                    }
                }

                return ret;

            }

            return ret;

        }
        private int FindRmax(int[] data1)
        {
            int ret = 0;
            int rMax = -10000;
            for (int i = 0; i < data1.Length; i++)
            {
                if (data1[i] > rMax)
                {
                    rMax = data1[i];
                    ret = i;
                }
            }
            data[col][(int)rowNums.rPeak] = rMax;

            data[col][(int)rowNums.rVal] = ret;

            return ret;
        }

        int FindQ(int start)
        {
            int ret = 0;
            int i;
            int qMin = 0; //this is only interesting if the Q is less than 0;
            int qVal = 0;
            //Step 1 Look back from the Peak of the R wave for the first 0 crossing.  \
            //this should be the peak of the Q wave.
            if (start < 40)
                return ret - 1;
            for (i = start - 5; i > start - 40; i--)
            {
                if (dArray[i] <= 0) break;

            }
            for (int j = i + 5; j > i - 5; j--)
                if (ECGarray[j] < qMin)
                {
                    qMin = ECGarray[j];
                    qVal = j;
                }
            for (i = i - 1; i > start - 40; i--)
            {
                if (d2Array[i] <= 0) break;

            }
            ret = i;
            data[col][(int)rowNums.qStart] = ret;
            data[col][(int)rowNums.qPeak] = qMin;
            data[col][(int)rowNums.qVal] = qVal;

            return ret;

        }

        int FindS(int start)
        {
            int ret = 0;
            int i;
            int sMin = 0;  //this is only interesting if S is less than 0.
            //Step 1 Look forward from the Peak of the R wave for the first 0 crossing.  \
            //this should be the peak of the S wave.

            for (i = start + 5; i < start + 40; i++)
            {
                if (dArray[i] >= 0) break;

            }
            data[col][(int)rowNums.sPeak] = ECGarray[i];

            //Step 2 Look forward for a change in the acceration of the derivative (i.e. third derivative of the ECG)
            //this should be the end of the Q wave
            if (i >= start + 40) return ret;
            int d3;
            for (i = i + 2; i < start + 40; i++)
            {
                d3 = d2Array[i] - d2Array[i - 1];

                if (d3 >= 0) break;

            }
            ret = i;
            data[col][(int)rowNums.sVal] = ret;

            return ret;

        }

        int FindPmax(int start)
        {
            int ret = 0;
            int end = start - 150;
            int peak = -10000;
            if (end < 0) end = 0;
            for (int i = start; i > end; i--)
            {
                if (ECGarray[i] > peak)
                {
                    peak = ECGarray[i];
                    ret = i;
                }

            }

            data[col][(int)rowNums.pPeak] = peak;
            data[col][(int)rowNums.pVal] = ret;
            return ret;

        }

        int FindP(int start)
        {
            int end = start - 50;
            if (end < 0) end = 0;
            int ret = 0;
            int peak = -10000;
            for (int i = start - 5; i > end; i--)
            {

                if (d2Array[i] > peak)
                {
                    ret = i;
                    peak = d2Array[i];
                }
            }
            data[col][(int)rowNums.pStart] = ret;
            return ret;

        }

        int FindTmax(int start, int q)
        {
            int ret = 0;
            int end = q + 250;
            int peak = -10000;
            if (end > ECGarray.Length) end = ECGarray.Length;
            for (int i = start; i < end; i++)
            {
                if (ECGarray[i] > peak)
                {
                    peak = ECGarray[i];
                    ret = i;
                }
            }
            data[col][(int)rowNums.tPeak] = peak;
            data[col][(int)rowNums.tVal] = ret;
            return ret;

        }

        int FindT(int start, int q)
        {
            int ret = 0;
            int end = q + 250;
            if (end > d2Array.Length) end = d2Array.Length;
            int peak = -10000;
            for (int i = start + 5; i < end; i++)
            {

                if (d2Array[i] > peak)
                {
                    ret = i;
                    peak = d2Array[i];
                }
            }
            data[col][(int)rowNums.tEnd] = ret;

            return ret;

        }
        int FindMinMax(int start)
        {
            int max = 0;
            if ((start - 2 < 0) || (start + 2 > ECGarray.Length))
                return max;

            for (int i = start - 5; i < start + 5; i++)
            {
                if (Math.Abs(ECGarray[i]) > max)
                    max = ECGarray[i];

            }
            return max;
        }

        int FindPlength(int start)
        {
            int end = start - 40;
            if (end < 0) end = 0;
            int begin, i;
            for (i = start; i > end; i--)
                if (ECGarray[i] <= 0) break;
            begin = i; //get the start
            end = start + 40;
            if (end > ECGarray.Length)
                end = ECGarray.Length;
            for (i = start; i < end; i++)
                if (ECGarray[i] <= 0) break;
            return i - begin;


        }


    }
}
