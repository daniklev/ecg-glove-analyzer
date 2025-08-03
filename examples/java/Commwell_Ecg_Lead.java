package commwellmedical.com.gloveecgservice;

import android.util.Log;

import java.util.LinkedList;

public class Commwell_Ecg_Lead {

    private final String LOG_TAG = "ECG12_SRV";

    private final double GAIN_1MV = 200;
    private final short BAD_VALUE = 9999;
    private final int SAMPLE_RATE = 500;
    private final static int MAX_RAW_DATA_SIZE = 15000;  //15000->   30  sec of data  sample 500 samp/sec

    private LinkedList<Short> row_lead = new LinkedList();//(MAX_RAW_DATA_SIZE);
    private int countBaseLine = 0;
    private boolean IsRecording = false;
    private int sensitivityLevel;
    private int preRecTime;
    private int recTime;
    private int powerLineFreqFilter;
    private boolean spike_removal_filter = true;
    //TODO  set to default value
    private float ecgRange = 1;  // default value 5
    private int filterType = 0;

    short[] rep_lead = new short[EcgDataView.REPORT_WIDTH];
    String leadName = "";
    String filteredData;

    public Commwell_Ecg_Lead() {
        countBaseLine = 0;
    }

    void SetPowerLineFreq(int _powerLineFreq) {
        powerLineFreqFilter = _powerLineFreq;
    }

    void AddNewVal(short newVal) {

        if (row_lead == null) {
            LogMe.e("spike", "Error New val: " + newVal + "row_lead=null");
            return;
        }

        // remove extra value from list
        if (!row_lead.isEmpty())
            if (row_lead.size() >= MAX_RAW_DATA_SIZE) //LogMe.w(LOG_TAG, "MADA ERROR size " + row_lead.size());
                row_lead.remove(); //TODO  ERROR MADA ->  java.util.NoSuchElementException

        //add new value to list
        row_lead.add(newVal);

        //Check if lead is out of range
        if (Math.abs(newVal) > GetTrashLevelBaseLine())
            countBaseLine = 0;
        else
            countBaseLine++;

    }

    public short GetLastRawValue() {
        return (short) row_lead.getLast();
    }

    void StartRecord(boolean rec) {
        IsRecording = rec;
    }

    void ClearAllData() {
        row_lead.clear();
        for (int i = 0; i < rep_lead.length - 1; i++)
            rep_lead[i] = 0;
        countBaseLine = 0;
        IsRecording = false;
        System.gc();
    }

    public int InBaseLineSec() {
        return (int) countBaseLine / SAMPLE_RATE;
    }

    public String GetLastRawData() {
        StringBuffer strb = new StringBuffer();
       /* for (short numb : row_lead)
            strb.append(numb + " ");*/

        int startIndx;
        Short[] rawData = row_lead.toArray(new Short[0]);
        short[] recData = new short[recTime * SAMPLE_RATE];
        short[] ftrData = null;

        if (rawData.length > recTime * SAMPLE_RATE)
            startIndx = rawData.length - recTime * SAMPLE_RATE - 2;
        else
            startIndx = 0;
        int endIndex = recTime * SAMPLE_RATE;
        if (rawData.length < endIndex)
            endIndex = rawData.length;

        try {
            for (int i = 0; i < endIndex - 1; i++) {
                if ((startIndx + i) < rawData.length - 1) {
                    short val = rawData[startIndx + i];
                    strb.append(val + " ");
                }
            }
        } catch (Exception e) {
            LogMe.e(LOG_TAG, "Error GetSavedFilteredData step 1.2 " + e.toString() + " " + e.getMessage() + ",destin: " + recData.length + ", source: " + rawData.length);
        }
        return strb.toString();
    }

    public String GetRowData() {
        StringBuffer strb = new StringBuffer();
        for (short numb : row_lead)
            strb.append(numb + " ");
        return strb.toString();
    }

    public String GetSavedFilteredData() {

        int startIndx;
        StringBuffer strb = new StringBuffer();
        Short[] rawData = row_lead.toArray(new Short[0]);
        short[] recData = new short[recTime * SAMPLE_RATE];
        short[] ftrData = null;

        if (rawData.length > recTime * SAMPLE_RATE)
            startIndx = rawData.length - recTime * SAMPLE_RATE - 2;
        else
            startIndx = 0;
        int endIndex = recTime * SAMPLE_RATE;
        if (rawData.length < endIndex)
            endIndex = rawData.length;
        try {
            for (int i = 0; i < endIndex - 1; i++) {
                if ((startIndx + i) < rawData.length - 1) {
                    short val = rawData[startIndx + i];
                    recData[i] = (short) (val * -1);
                }
            }
        } catch (Exception e) {
            LogMe.e(LOG_TAG, "Error GetSavedFilteredData step 1.2 " + e.toString() + " " + e.getMessage() + ",destin: " + recData.length + ", source: " + rawData.length);
        }
        try {
            ftrData = Filter5060Hz(recData, powerLineFreqFilter);
        } catch (Exception e) {
            LogMe.e(LOG_TAG, "Error GetSavedFilteredData step 1.3 " + e.toString());
        }

        //For  morthologic filter
        if (filterType == 0) {
            try {
                MorphologyFilterClass morfFilter = new MorphologyFilterClass();
                for (int i = 0; i < endIndex; i++) {
                    if (ftrData[i] != BAD_VALUE)
                        ftrData[i] = (short) morfFilter.computeHPF(ftrData[i]);
                    else
                        ftrData[i] = BAD_VALUE;
                }
            } catch (Exception e) {
                LogMe.e(LOG_TAG, "Error GetSavedFilteredData step 1.4 " + e.toString());
            }
        }

        //For  Hi pass filter 0.15
        if (filterType == 1) {
            try {
                HiPassFilter hpfFilter = new HiPassFilter(HP_FILETER_TYPE.HP015);
                for (int i = 0; i < endIndex; i++) {
                    if (ftrData[i] != BAD_VALUE)
                        ftrData[i] = (short) hpfFilter.GetNewVal(ftrData[i]);
                    else
                        ftrData[i] = BAD_VALUE;
                }
            } catch (Exception e) {
                LogMe.e(LOG_TAG, "Error GetSavedFilteredData step 1.4 " + e.toString());
            }
        }

        if (ftrData == null)
            return "";


        PrepareLeadForReport(ftrData, filterType == 0 ? spike_removal_filter : false);

        for (short numb : recData)
            strb.append(numb + " ");
        strb.append('\n');

        filteredData = strb.toString();
        return filteredData;
    }

    private short[] Filter5060Hz(short[] data, int filtr) {
        if (filtr == 0)
            return data;
        short[] filteredData = new short[data.length];
        NotchEcgFilter nf = new NotchEcgFilter();
        nf.InitNotchClass(filtr);

        for (int i = 0; i < data.length - 1; i++) {
            short val = data[i];
            if (val != BAD_VALUE)
                filteredData[i] = (short) Math.round(nf.GetNewVal(val));
            else
                filteredData[i] = BAD_VALUE;
        }
        //todo enable 100/120 hx filter ( now is disabled -  add noise to signal)
       /*
        NotchEcgFilter nf2 = new NotchEcgFilter();
        nf2.InitNotchClass(filtr*2);
        for (int i = 0; i < data.length - 1; i++)
            filteredData[i] =  (short) Math.round(nf2.GetNewVal(data[i]));
            */
        return filteredData;
    }

    private void PrepareLeadForReport(short[] filtrRawData, boolean remove_spike) {

        short[] filtrData = new short[0];
        if (remove_spike)
            filtrData = SpikeRemoveFilter.FilterSpikeDataData(filtrRawData);
        else
            filtrData = filtrRawData;

        LogMe.d(LOG_TAG, "SpikeRemoveFilter run" + filtrData.length);

        final double GAIN_1MV_FACTOR = ((EcgDataView.REPORT_WIDTH / 10) / 5) * 2 / GAIN_1MV;

        int resampleDividerCounter = 0;
        int resampleDivider2Counter = 0;
        int currReportDataPosition = 0;

        int currRawEcgDataPosition = 500; // Shift 1 sec for HPF - remove slope

        while ((currRawEcgDataPosition < filtrData.length - 2) && (currReportDataPosition < rep_lead.length - 2)) {
            resampleDividerCounter++;
            resampleDivider2Counter++;
            if (resampleDivider2Counter >= 18) {
                currRawEcgDataPosition++;
                resampleDivider2Counter = 0;
            }
            if ((resampleDividerCounter == 3) || (resampleDividerCounter == 6) || (resampleDividerCounter == 9))
                currRawEcgDataPosition++;
            else
                rep_lead[currReportDataPosition++] = (short) Math.round(filtrData[currRawEcgDataPosition++] * GAIN_1MV_FACTOR);
            if (resampleDividerCounter >= 10)
                resampleDividerCounter = 0;
        }
    }

    public int GetTrashLevelBaseLine() {
        return (int) Math.round(GAIN_1MV * ecgRange);
    }

    void setSensitivityLevel(int _sensitivityLevel) {
        sensitivityLevel = _sensitivityLevel;
    }

    void setEcgRange(float _ecgRange) {
        ecgRange = _ecgRange;
    }

    void setFilterType(int _filterType) {
        filterType = _filterType;
    }


    void setPreRecTime(int _preRecTime) {
        preRecTime = _preRecTime;
    }

    void setRecTime(int _recTime) {
        recTime = _recTime;
    }

    void setSpike_removal_filter(boolean _spike_removal_filter) {
        spike_removal_filter = _spike_removal_filter;
    }

}
