using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSProject
{
    public class Triple
    {
        double[] first, second, last;

        public Triple(double[] x)
        {
            first = x;

            second = new double[x.Length];
            Array.Fill(second, 0);

            last = (double[])second.Clone();
        }

        public Triple(double[] x, double[] y, double[] z)
        {
            first = x;
            second = y;
            last = z;
        }

        public double[] First()
        {
            return first;
        }

        public double[] Second()
        {
            return second;
        }
        public double[] Last()
        {
            return last;
        }
    }
}
