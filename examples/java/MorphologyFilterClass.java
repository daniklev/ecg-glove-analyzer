package commwellmedical.com.gloveecgservice;

/* Created by Alex Levshin on 28/08/2015.
    Unit : UtilsClass
    Finished:
    Status: Not Complete
    Version: 1.0
    Description:
    FD :
    Bugs :
    Updates : xamarin
    */


class Buffer {
    int[] buffer;
    int start;
    int size;
    Boolean full;
    int filling;
    int sum;
    int count;

    public Buffer(int buffersize) {
        buffer = new int[buffersize];
        size = buffersize;
        full = false;
        filling = 0;
    }

    public Boolean Fill() {
        return filling == size ? true : false;
    }

    public void Add(int x) {
        if (filling == size) {
            sum = sum - buffer[start];
            if (buffer[start] != 0) count--;
            sum = sum + x;
            buffer[start++] = x;
            if (x != 0) count++;
        } else {
            filling++;
            buffer[start++] = x;
            sum = sum + x;
            if (x != 0) count++;
        }
        if (start >= size) start = 0;
    }

    public int GetCount() {
        return count;
    }

    public int GetAvg() {
        if (count == 0)
            return 0;
        else
            return sum / count;
    }

    public int GetStart() {
        return buffer[start];
    }

}

public class MorphologyFilterClass {
    int start = 0;
    int InputBufferSize = 150;
    Buffer input = new Buffer(InputBufferSize);
    int AvgBufferSize = 250;
    Buffer Avg1 = new Buffer(AvgBufferSize);
    int LPFBufferSize = 10;
    int DelayBufferSize = (InputBufferSize + AvgBufferSize + LPFBufferSize) / 2;
    Buffer delay = new Buffer(DelayBufferSize);
    Buffer LPF = new Buffer(LPFBufferSize);
    int OldData;
    int DeltaThreshold = 5;

    public int computeHPF(int data) {
        int value = 0;
        delay.Add(data);
        LPF.Add(data);
        if (Math.abs(data - OldData) < DeltaThreshold)
            input.Add(LPF.GetAvg());
        else
            input.Add(0);
        OldData = data;
        if (input.Fill()) {
            Avg1.Add(input.GetAvg());
            if (Avg1.Fill())
                value = delay.GetStart() - Avg1.GetAvg();
        }
        return value;
    }

}
