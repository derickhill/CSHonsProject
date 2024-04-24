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

        public double[] First { get { return first; } }

        public double[] Second { get { return second; } }
        public double[] Last { get { return last; } }
    }
}
