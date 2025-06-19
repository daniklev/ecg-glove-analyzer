using System;
using System.Collections.Concurrent;
//using ECGanalysis;
using System.Diagnostics;

namespace CommwellEcgCalculation
{
    class QRS1
    {
        const string version = "Version 1.0.0.0";
        const int sampleSize = 10000 * 3;
        int[,] marker1 = new int[600, 2]; //QRS marker marker[x,0] is the file position, marker[x,1] is the beat type
        int[,] marker2 = new int[600, 2];
        public int[,] marker = new int[600, 2];
        byte[] f10 = new byte[1000000];  //stores file data
        int[,] annotation1 = new int[600, 2]; //annotation[x,0] is the sample time of annotation
        public int[] f3 { get; set; }
        public int[] outValue1 { get; set; }
        public int[] outValue2 { get; set; }
        public int[] f4;

        public int[] f5;
        public int[] f6;
        public int[] markers;
        //annotation[x,1] is the annotation code for lead 1
        ConcurrentQueue<int> times = new ConcurrentQueue<int>();
        string file;        //holds file name without extention
        const int sweep = 1;
        string record;
        int samples = 500;
        int dataSize;
        public int rrCnt1;
        public int rrCnt2;
        public ConcurrentQueue<int> rr = new ConcurrentQueue<int>();
        int rrCnt;
        int annotationPosition = 0;
        ConcurrentQueue<int> Noises = new ConcurrentQueue<int>();
        //int lastRR;
        UInt32 last_packet_number = 0;
        // BDAC compute = new BDAC();
        // BDAC compute1 = new BDAC();
        int dataStart;
        int data1Start;
        int displayCnt = 0x8000;
        int tachyDisplayUpdate = 4;
        int bradyDisplayUpdate = 4;
        int afibDisplatUpdate = 4;

        /* OUTPUT QUEUES */
        ConcurrentQueue<int[]> f3Out = new ConcurrentQueue<int[]>();
        ConcurrentQueue<int[]> f4Out = new ConcurrentQueue<int[]>();
        ConcurrentQueue<int[,]> markerOut = new ConcurrentQueue<int[,]>();
        ConcurrentQueue<int[,]> annotationsOut = new ConcurrentQueue<int[,]>();
        ConcurrentQueue<int> markerOutSize = new ConcurrentQueue<int>();
        ConcurrentQueue<int> stripOffset = new ConcurrentQueue<int>();
        int[,] sympton = new int[50, 2];
        int symptonCount = 0;
        int timeout = 1000;
        //FFT fft = new FFT();

        static double[] bHP1 = new double[5] { 0.167179268608490, -0.668717074433960, 1.003075611650939, -0.668717074433960, 0.167179268608490 };
        static double[] aHP1 = new double[5] { 1.0000000000000000, -0.7820951980233374, 0.6799785269162991, -0.1826756977530322, 0.0301188750431692 };
        static double[] bHP2 = new double[5] { 9.918242120005331e-01, -3.967296848002132e+00, 5.950945272003199e+00, -3.967296848002132e+00, 9.918242120005331e-01 };
        static double[] aHP2 = new double[5] { 1.000000000000000e+00, -3.983581258658520e+00, 5.950878429266698e+00, -3.951012436572833e+00, 9.837152675104786e-01 };
        static double[] bHP8hz = new double[6] { 0.8498059442850544, -4.249029721425272, 8.498059442850543, -8.498059442850543, 4.249029721425272, -0.8498059442850544 };
        static double[] aHP8hz = new double[6] { 1.000000000000000, -4.674714813483247, 8.751137525665719, -8.200005770674688, 3.845761964355956, -0.7221701429421383 };
        static double[] bLP16hz = new double[6] { 7.537890968216628e-06, 3.768945484108314e-05, 7.537890968216629e-05, 7.537890968216629e-05, 3.768945484108314e-05, 7.537890968216628e-06 };
        static double[] aLP16hz = new double[6] { 1.000000000000000e+00, -4.349665772068885e+00, 7.605137812388491e+00, -6.678045189042118e+00, 2.943812259738306e+00, -5.209978985048100e-01 };
        static double[] bDeriv1 = new double[2] { -1, 1 };
        static double[] bLP1 = new double[25];
        filter HP1 = new filter(bHP1, aHP1, 5);
        filter Deriv1 = new filter(bDeriv1, 2);
        filter LP1 = new filter(50);
        filter HP2 = new filter(bHP1, aHP1, 5);
        filter Deriv2 = new filter(bDeriv1, 2);
        filter LP2 = new filter(50);
        filter HP1_05hz = new filter(bHP2, aHP2, 5);
        filter HP2_05hz = new filter(bHP2, aHP2, 5);
        filter HP3_05hz = new filter(bHP2, aHP2, 5);
        filter HP4_05hz = new filter(bHP2, aHP2, 5);
        filter HP8hz = new filter(bHP8hz, aHP8hz, 6);
        filter HP16hz = new filter(bLP16hz, aLP16hz, 6);
        private static object _syncRoot2 = new object();
        private static object _syncRoot3 = new object();
        private static object _syncRoot4 = new object();

        bool noisyData = false;
        int totalNoise;
        int[] avgBeatArray = new int[600];
        int[] avgVentArray = new int[600];
        int[] avgVent1Array = new int[600];
        int[] rrStatistics = new int[600];
        double[] correlarray = new double[600];
        double afibmax;
        int[] beat1 = new int[300];
        int[] beat2 = new int[300];
        int[] beat3 = new int[300];
        int[] beat4 = new int[300];

        int[] avgBeatArray1 = new int[600];
        int[] rrStatistics1 = new int[600];
        double[] correlarray1 = new double[600];
        int HRmean;
        int HRsigma;
        int HRmean1;
        int HRsigma1;
        int HRmax;
        int HRmax1;
        bool afStart = false;
        bool waitingforAFstart = true;
        bool bradyStart = false;
        bool waitingforBradystart = true;
        bool tachyStart = false;
        bool waitingforTachystart = true;
        bool pauseStart = false;
        bool waitingforPausestart = true;
        bool rhythmStart = false;
        int ventCnt = 0;
        int runCnt = 0;

        int Rmax = 0;
        bool Rpositive = true;
        int Rtime = 0;
        int RbeatCnt = 0;
        int R1max = 0;
        int Rmaxtest = 0;
        int Rtimetest = 0;
        bool R1positive = true;
        int R1time = 0;
        int R1beatCnt = 0;
        int Rwidth;
        int R1width;
        int VbeatCnt = 0;
        int Vmax = 0;
        int VbeatTime = 0;
        int VbeatWidth = 0;
        int V1beatCnt = 0;
        int V1beatTime = 0;
        int V1beatWidth = 0;
        int V1max = 0;
        bool gain1 = false;
        bool gain2 = false;
        bool[] p_exists = new bool[600];
        bool allowtesting = false;
        char[] intervals = new char[600];
        int[] HRs = new int[600];
        bool pvcFound = false; //only show one PVC per strip
        bool afFound = false; //don't show both rhythm change and AF on same strip
        private Morph[] morph = new Morph[8];
        public int[] morphCount = new int[8];
        public int[] morphOrder = new int[8];
        private Morph[] fmorph = new Morph[8];
        private Morph[] gmorph = new Morph[8];
        private int[][] waves = new int[8][];
        public int[] outWave1 = new int[400];
        public int[] outWave2 = new int[400];
        public int[] outWave3 = new int[400];
        public int[] outWave4 = new int[400];
        ConcurrentQueue<int> lead1 = new ConcurrentQueue<int>();
        ConcurrentQueue<int> lead2 = new ConcurrentQueue<int>();
        public ConcurrentQueue<int>[] leads = new ConcurrentQueue<int>[8];

        public QRS1()
        {

        }
        public int Glove(byte[] glove)
        {
            int ret = 0;
            for (int i = 0; i < 8; i++)
                leads[i] = new ConcurrentQueue<int>();
            ret = DecodeGlove(glove);
            lead1 = leads[0];
            lead2 = leads[1];
            ret = ECG(lead1.ToArray(), lead2.ToArray());

            return ret;
        }

        private int DecodeGlove(byte[] glove)
        {
            int ret = 0;
            int size = glove.Length;
            int packetType = 0;

            for (int i = 0; i < size;)
            {
                while (glove[i] != 0x80)
                {
                    i++;
                    if (i > size - 11) return -1;
                }
                if (i > size - 11) return -1;
                if ((glove[i + 1] == 0x17) && (glove[i + 2] == 0))
                    packetType = checkHeader(glove, i);
                if (packetType == 0x51)
                {
                    i += 7;
                    if ((i + 81) < size)
                        decodeGloveECG(glove, i);
                    i += 81;
                }
                else if (packetType == 3)
                {
                    i += 1;
                    decodeGloveFault(glove, i);
                    i += 10;
                }
                else i++;
            }


            return ret;
        }
        private void computeLead(byte[] glove, int start, int index)
        {
            Int16 temp;
            start += index * 2;
            temp = glove[start + 1];
            temp *= 256;
            temp += glove[start];
            leads[index].Enqueue(temp);
        }
        private int decodeGloveECG(byte[] glove, int start)
        {
            int ret = 0;
            byte cs = 0;
            for (int i = start; i < (start + 81); i++)
                cs += glove[i];
            if (cs == 0)
                for (int i = start; i < start + 80; i += 16)
                    for (int j = 0; j < 8; j++)
                        computeLead(glove, i, j);
            else
                ret = -1;
            return ret;
        }

        private int decodeGloveFault(byte[] glove, int start)
        {
            int ret = 0;
            return ret;
        }
        private int checkHeader(byte[] glove, int start)
        {
            int ret = 0;
            byte cs = 0;
            for (int i = 0; i < 7; i++)
                cs += glove[start + i];
            if (cs == 0)
                ret = glove[start + 5];
            return ret;
        }
        public int ECG(int[] leadI, int[] leadII)
        {
            dataSize = leadI.Length;
            f3 = new int[dataSize];
            f3 = leadI;

            f4 = new int[dataSize];
            f4 = leadII;

            f5 = new int[dataSize];
            f6 = new int[dataSize];
            double temp1, temp2;
            for (int i = 0; i < 200; i++)
            {
                HP1_05hz.filterValue(f3[i]);
                HP2_05hz.filterValue(f4[i]);
            }
            for (int i = 0; i < dataSize; i++)
            {
                //temp1 = HP16hz.filterValue(f3[i]);
                temp1 = HP1_05hz.filterValue(f3[i]);
                //temp2 = HP16hz.filterValue(f4[i]);
                temp2 = HP2_05hz.filterValue(f4[i]);
                temp1 = (temp1 * 0.8); //lead I
                temp2 = (temp2 + 0.1 * temp1);  //lead III
                f4[i] = (int)temp2;
                temp2 = temp1 / 2 + temp2; // lead Avf
                f3[i] = (int)temp1;
                f5[i] = (int)temp2;

            }
            myleads(leads[4].ToArray(), leads[2].ToArray());

            int tempSize = 0;
            dataSize = f3.Length;
            outValue1 = new int[dataSize];
            outValue2 = new int[dataSize];
            Stopwatch mytime = new Stopwatch();
            int irving;
            //if (!Monitor.TryEnter(_syncRoot3, timeout))
            //throw new TimeoutException();
            //
            // First Step decode data from 3 consecutive files
            // Each file containing about 32 seconds of data.
            // Only the middle file will be reported
            //
            //try
            {

                // First pass is to find the QRS complexes for lead 1
                try
                {
                    rrCnt1 = thresholdDetector(marker1, f3, outValue1, out outWave1);

                }
                catch (Exception e)
                {
                    return -101;
                }


                try
                {
                    rrCnt2 = thresholdDetector(marker2, f4, outValue2, out outWave2);

                }
                catch (Exception e)
                {
                    return -104;
                }
                if ((rrCnt1 == 0) && (rrCnt2 == 0))
                    return -200;
                if (rrCnt1 > rrCnt2)
                {
                    markers = outValue1;
                    marker = marker1;
                    rrCnt = rrCnt1;
                }
                else
                {
                    markers = outValue2;
                    marker = marker2;
                    rrCnt = rrCnt2;
                }
                for (int i = 1; i < rrCnt; i++)
                    rr.Enqueue(marker[i, 0] - marker[i - 1, 0]);
                for (int i = 0; i < outWave3.Length; i++)
                {
                    outWave3[i] = outWave1[i] / 2 + outWave2[i];
                    outWave4[i] = outWave1[i] + outWave2[i];
                }
                mytime.Stop();
                Console.WriteLine("Time elapsed: {0}", mytime.Elapsed);
            }
            //finally { Monitor.Exit(_syncRoot3); }
            return 0;

        }

        private void QRS(int[] fin, int[] fout)
        {
            for (int i = 0; i < dataSize - 16; i++)
            {
                fout[i] = (int)HP8hz.filterValue(fin[i]);
                fout[i] = (int)(HP16hz.filterValue(fout[i]) * 10);
                fout[i] = Math.Abs(fout[i + 1] - fout[i]);
            }
            for (int i = 0; i < dataSize - 16; i++)
                for (int ii = 0; ii < 16; ii++) fout[ii] += fout[i + ii];
        }

        /// <summary>
        /// Threshold detector on the derivitive of the data
        /// </summary>
        /// <param name="data"></param> holds the time and type of the QRS complex
        /// <param name="fin"></param> holds the derivative of the filtered signal signal
        /// <param name="truef"></param> hold the filtered signal system gain is 73 digital units = 1 mV
        /// reject signals whose values are less than 0.15mV = ll bits or greater than 5mV = 365 bits
        /// <returns></returns>

        private int thresholdDetector(int[,] data, int[] fdata, int[] outValues, out int[] outWave)
        {
            int lasthold = 200;
            int threshold = 200;  //set to minimum threshold
            int peakValue = 0;
            int peakTime = 0;
            int peaks = 0;
            bool found = false;
            int cnt = 0;
            int peakcnt = 0;
            int nobeatcount = 0;
            int longtermpeak = 0;
            int longtermcnt = 0;
            int truepeak;
            int truepeaktime = 0;
            bool retest = false;
            int qrsTimer = 0;
            int dataSize = fdata.Length;
            int[] fin = new int[dataSize];
            QRS(fdata, fin);
            for (int i = 0; i < morph.Length; i++)
            {
                morph[i] = new Morph(200);
                fmorph[i] = new Morph(200);
                gmorph[i] = new Morph(400);
                morphCount[i] = 0;

            }

            for (int i = 75; i < dataSize - 16; i++)
            {
                if ((fin[i] > threshold) && (i < peakTime + 100)) // 40 -> 500/200 = 10/4 = 2.5 -> 100 
                {
                    peakTime = i;
                    peakValue = fin[i];
                    threshold = peakValue;
                    found = true;
                    nobeatcount = 0;
                }
                if (!found)
                {
                    peakTime = i;
                    //if more than 3 seconds without beats reset threshold to minimumm
                    if (nobeatcount++ > 1500)
                    {
                        if (!retest)  //only retest once so we are not in an endless loop
                        {
                            i = i - nobeatcount; //go back and retest;
                            if (i < 100) i = 100;
                            retest = true;
                        }
                        lasthold = 100;
                        nobeatcount = 0;
                        threshold = 200;

                    }

                }
                else if ((i > peakTime + 100) && found)
                {
                    int[] val = new int[200];
                    int[] val1 = new int[400];
                    int max = 0;
                    int relTime = peakTime - 100 - 50;
                    int morphClass = 0;
                    if ((relTime > 100) && (relTime + 400 < fin.Length))
                    {
                        for (int j = 0; j < 200; j++)
                            val[j] = (int)HP1_05hz.filterValue(fdata[relTime + j]);
                        morphClass = checkMorph(val);
                        max = FindRmax(val);
                    }
                    if (max > 50)
                    {
                        if (morphClass < fmorph.Length)
                        {
                            fmorph[morphClass].addWave(val);
                            for (int j = 0; j < 400; j++)
                                val1[j] = (int)HP1_05hz.filterValue(fdata[relTime - 50 + j]);
                            gmorph[morphClass].addWave(val1);

                        }

                        i = peakTime + 125; //start looking for a new peak  72/// ile changed for testing
                        found = false;
                        retest = false;
                        truepeak = 0;
                        truepeaktime = peakTime;
                        for (int j = peakTime; j > peakTime - 100; j--)
                        {
                            if (fin[j] > truepeak)  ///fin
                            {
                                truepeak = fin[j];
                                truepeaktime = j;
                            }

                        }
                        if (truepeak == 0)  //must be a negative going peak
                            for (int j = peakTime; j > peakTime - 100; j--)
                            {
                                if (fin[j] < truepeak)
                                {
                                    truepeak = fin[j];
                                    truepeaktime = j;
                                }

                            }
                        if (fin[truepeaktime] > 50) //only count peaks greater than 0.15mV
                        {
                            if (truepeak != 0) data[cnt, 0] = truepeaktime;
                            else data[cnt, 0] = peakTime;
                            if (peakValue < 5000)
                            {
                                qrsTimer = 25;
                                data[cnt++, 1] = 1;
                            }
                            else data[cnt++, 1] = 0; //this is most likely noise but still need to check
                        }
                        peaks += peakValue;
                        peakcnt++;
                        longtermpeak += peakValue;
                        longtermcnt++;
                        if (peakcnt == 8)
                        {
                            lasthold = peaks / 16;  //one third the average peak
                            peakcnt = 0;
                            peaks = 0;
                            if (lasthold > longtermpeak / longtermcnt)
                                lasthold = longtermpeak / longtermcnt / 3;
                        }

                        threshold = 1 * threshold / 2;

                    }
                    else

                        found = false;


                }
                else
                {

                }

                if (qrsTimer > 0)
                {
                    qrsTimer--;
                    if ((truepeaktime + qrsTimer - 50) > 0) outValues[truepeaktime + qrsTimer - 50] = -200;
                }
                else outValues[i] = 0;

            }
            checkMorphClass();
            reorderMorphClass();
            outWave = gmorph[morphOrder[0]].averageWave();
            /*for (int i = 0; i < gmorph[morphOrder[0]].size; i++)
                outWave[i] = gmorph[morphOrder[0]].wave[i];
            */
            return gmorph[morphOrder[0]].numberOfWaves;
        }

        private void checkMorphClass()
        {
            for (int i = 0; i < morphCount.Length; i++)
            {
                morph[i].validateMorph();
                morphCount[i] = morph[i].numberOfWaves;

            }

        }
        private void reorderMorphClass()
        {
            int[] myKeys = new int[morphCount.Length];
            int[] myValues = new int[morphCount.Length];
            for (int i = 0; i < morphCount.Length; i++)
            {
                myKeys[i] = morphCount[i];
                myValues[i] = i;
            }
            Array.Sort(myKeys, myValues);
            for (int i = 0; i < morphCount.Length; i++) morphOrder[morphCount.Length - i - 1] = myValues[i];

        }
        private int checkMorph(int[] data)
        {
            double thresh = 0.85;
            bool finished = false;
            int i = 0;
            while (!finished)
            {
                if (morph[i].numberOfWaves == 0)
                {
                    finished = true;
                }
                else if (morph[i].compareWave(data) > thresh)
                {
                    finished = true;
                }
                else i++;

                if (i == morphCount.Length) finished = true;
            }

            if (i < morphCount.Length)
            {
                morphCount[i]++;
                morph[i].addWave(data);
            }
            return i;
        }

        private int FindRmax(int[] data)
        {
            int ret = 0;
            int rMax = -10000;
            for (int i = 0; i < data.Length; i++)
            {
                if (Math.Abs(data[i]) > rMax)
                {
                    rMax = Math.Abs(data[i]);
                    ret = i;
                }
            }


            return rMax;
        }

        private void myleads(int[] leada, int[] leadb)
        {
            int temp1;
            int temp2;
            for (int i = 0; i < leada.Length; i++)
            {
                temp1 = (int)HP3_05hz.filterValue(leada[i]);
                temp2 = (int)HP4_05hz.filterValue(leadb[i]);
                f6[i] = temp2 - temp1;
            }
        }
        public class filter
        {
            int size;
            double[] dataArray = new double[100];
            double[] yArray = new double[100];
            double[] b = new double[100];
            double[] a = new double[100];

            public filter(double[] b1, int size1)   //for FIR filter
            {
                size = size1;
                for (int i = 0; i < size; i++)
                {
                    b[i] = b1[i];
                    a[i] = 0;
                }
                a[0] = 1;
            }
            public filter(int size1)  //for simple average
            {
                size = size1;
                for (int i = 0; i < size; i++)
                {
                    b[i] = 1.0 / size;
                    a[i] = 0;
                }
            }
            public filter(double[] b1, double[] a1, int size1)  //for IIR filter
            {
                size = size1;
                for (int i = 0; i < size; i++)
                {
                    b[i] = b1[i];
                    a[i] = a1[i];
                }
            }

            public double filterValue(int newData)
            {
                double sum = 0.0;
                for (int i = size - 1; i > 0; i--)
                {
                    dataArray[i] = dataArray[i - 1];
                    sum += dataArray[i] * b[i];
                }

                dataArray[0] = newData;
                sum += newData * b[0];
                for (int i = size - 1; i > 0; i--)
                {
                    yArray[i] = yArray[i - 1];
                    sum -= yArray[i] * a[i];
                }
                yArray[0] = sum;

                return sum;

            }
        }



    }
}

