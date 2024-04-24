using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSProject
{
    public class Pair
    {
        double[] first, last;

        public Pair(double[] x)
        {
            first = x;
            last = new double[x.Length];
            Array.Fill(last, 0);
        }

        public Pair(double[] x, double[] y)
        {
            first = x;
            last = y;
        }

        public double[] First()
        { 
            return first; 
        }
        public double[] Last()
        { 
            return last;
        }
    }
}
