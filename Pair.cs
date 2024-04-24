using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSProject
{
    class Pair
    {
        double[] first, last;

        public Pair(double[] x)
        {
            first = x;
            last = new double[x.Length];
            Array.Fill(last, 0);
        }

        public double[] First { get { return first; } }
        public double[] Last { get { return last; } }
    }
}
