package commwellmedical.com.gloveecgservice;

/**
 * Created by Programmer01 on 21/12/2015.
 * xamarin
 */
enum HP_FILETER_TYPE { HP005 , HP015, HP05  } ;


public class HiPassFilter {


    private double[]  xv  = new  double[4];
    private double[] yv = new double[4];
    private double HP0;
    private double HP1;
    private double GAIN;
    private int NPOLES;


    public HiPassFilter(HP_FILETER_TYPE ftr_type)
    {
        //
        // TODO: Add constructor logic here
        //
        switch (ftr_type)
        {
            case HP05 :
                HP0 =  -0.9878018507 ;
                HP1 =   1.9877269954 ;
                GAIN = 1.006155446 ;
                NPOLES = 2;
                break;

            case HP015:
                HP0 =  -0.9963349287 ;
                HP1 =   1.9963282000 ;
                GAIN =  1.001837588 ;
                NPOLES = 2;
                break;

            case HP005:
                HP0 =  -0.9987734371 ;
                HP1 =   1.9987726844 ;
                GAIN =  1.00061384 ;
                NPOLES = 2;
                break;
        }
    }


    private void ClearValues()
    {
        for(int i = 0;i<NPOLES;i++)
        {
            xv[i] = 0;
            yv[i] = 0;
        }
    }

    public void InsertNewVal(double val)
    {
        xv[0] = xv[1];
        xv[1] = xv[2];
        xv[2] = val / GAIN ;
    }

    public double GetNewVal(double val )
    {
        InsertNewVal( val );

        yv[0] = yv[1];
        yv[1] = yv[2];
        yv[2] = (xv[0] +xv[2]) - 2 * xv[1]  + (HP0*yv[0]) + (HP1*yv[1]) ;

        return  yv[2] ;
    }

}
