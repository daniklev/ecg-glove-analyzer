using System.Collections.Concurrent;
namespace CommwellEcgCalculation
{

    public class CommCalc
    {
        QRS1 qrs = new QRS1();
        private int[] calc_values;
        private ConcurrentQueue<int>[] calc_leads;
        public string version = "V1.0.0.1";
        public int gloveCalc(byte[] gloveData)
        {
            ConcurrentQueue<int> numbers = new ConcurrentQueue<int>();
            qrs.Glove(gloveData);
            int[] wave1 = qrs.outWave1;
            int[] wave2 = qrs.outWave2;
            int[] wave3 = qrs.outWave3;
            int rr = 0;
            int cnt = qrs.rr.Count;
            if ((qrs.rrCnt1 < 4) || (qrs.rrCnt2 < 4))
            {
                return -1;
            }
            int sum = 0;
            int cntr = 0;
            int temp;

            int[] rrList = qrs.rr.ToArray();
            int min = 10000;
            int max = -1;

            cntr = rrList.Length; // total numbers of  RR intervals
            for (int j = 0; j < cntr; j++)
            {
                temp = rrList[j];
                temp += rrList[j];
                if (temp < min) min = temp;  // save min RR intervals
                if (temp > max) max = temp;  // save max RR intervals               
                sum += temp; // calc sum for all RR intervals
            }
            // calc average RR interval w/o max and min values
            rr = (sum - min - max) / (cntr - 2);

            //remove by Alex Original Irving code
            //rr = sum / cntr - min - max;
            //rr = rr * 2;            
            Calcs calcs = new Calcs(wave1, wave2, wave3, rr);
            calc_values = calcs.val;
            calc_leads = qrs.leads;
            return 0;

        }

        public int[] getValues()
        {
            return calc_values;
        }

        public ConcurrentQueue<int>[] getWaves()
        {
            return calc_leads;
        }


    }
}
