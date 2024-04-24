using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSProject
{
    class Tests
    {
        public static void Main(string[] args)
        {
            int maxFunctionCalls = 50000;

            for(int i = 0; i < 15; i++)
            {
                FitnessFunc fitnessFunc = new FitnessFunc(0, maxFunctionCalls);
                int dim = 10;
                int n = 100;

                double[][] pop = Algorithms.initialisePopulation(fitnessFunc.getLowerBound(dim), fitnessFunc.getUpperBound(dim), n);

                Algorithms.GradientDescentGA(pop, fitnessFunc, n);
            }
        }
    }
}
