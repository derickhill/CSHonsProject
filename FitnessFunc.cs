using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSProject
{
    public class FitnessFunc
    {
        public int count;
        double maxFitness;
        Func<double[], double> fitnessFunc;
        double lowerBound, upperBound;
        double[] functionCalls;
        int maxFunctionCalls;

        Func<double[], double>[] fitnessFuncs = [Ackley1Function, Schwefel1_2Function, Alpine1Function, Alpine2Function, Schwefel2_20Function, SphereFunction, Step3Function, ZakharovFunction, XinSheYangFunction, Trigonometric1Function];

        double[] upperBounds = [35, 100, 10, 10, 100, 10, 100, 10, 5, Math.PI];
        double[] lowerBounds = [-35, -100, -10, 0, -100, 0, -100, -5, -5, 0];

        public FitnessFunc(int i, int maxFunctionCalls)
        {
            fitnessFunc = fitnessFuncs[i];
            count = 0;
            maxFitness = double.MaxValue;
            this.maxFunctionCalls = maxFunctionCalls;
            functionCalls = new double[maxFunctionCalls];

            lowerBound = lowerBounds[i];
            upperBound = upperBounds[i];
        }

        public void reset()
        {
            count = 0;
            maxFitness = double.MaxValue;
            functionCalls = new double[maxFunctionCalls];
        }

        public int getMaxFunctionCalls()
        {
            return maxFunctionCalls;
        }

        public double[] getFunctionCalls()
        {
            return functionCalls;
        }

        public double evaluate(double[] x)
        {
            count++;

            double fitness = fitnessFunc(x);

            if(fitness < maxFitness)
                maxFitness = fitness;

            if(count <= maxFunctionCalls)
                functionCalls[count - 1] = maxFitness;

            return fitness;
        }

        public double getMaxFitness()
        {
            return maxFitness;
        }

        public int getCount()
        {
            return count;
        }

        public double[] getBest(double[][] population, int populationSize)
        {
            double minFitness = double.MaxValue;
            double[] best = population[0];

            for(int i = 0; i < populationSize; i++)
            {
                double curFitness = evaluate(population[i]);

                if(curFitness < minFitness)
                {
                    minFitness = curFitness;
                    best = population[i];
                }
            }

            return best;
        }

        public Pair getBest(Pair[] population, int populationSize)
        {
            double minFitness = double.MaxValue;
            Pair best = population[0];

            for (int i = 0; i < populationSize; i++)
            {
                double curFitness = evaluate(population[i].First());

                if (curFitness < minFitness)
                {
                    minFitness = curFitness;
                    best = population[i];
                }
            }

            return best;
        }

        public Triple getBest(Triple[] population, int populationSize)
        {
            double minFitness = double.MaxValue;
            Triple best = population[0];

            for (int i = 0; i < populationSize; i++)
            {
                double curFitness = evaluate(population[i].First());

                if (curFitness < minFitness)
                {
                    minFitness = curFitness;
                    best = population[i];
                }
            }

            return best;
        }

        public double[] getLowerBound(int dim)
        {
            double[] lb = new double[dim];
            Array.Fill(lb, lowerBound);

            return lb;
        }
        public double[] getUpperBound(int dim)
        {
            double[] ub = new double[dim];
            Array.Fill(ub, upperBound);

            return ub;
        }

        // -35 <= xi <= 35
        // min at (0, 0, 0, ... , 0) and is 0
        private static double Ackley1Function(double[] x)
        {
            return -20 * Math.Exp(-0.02 * Math.Sqrt((1 / (x.Length * 1.0)) * x.Sum(x_i => x_i * x_i))) - Math.Exp((1 / (x.Length * 1.0)) * x.Sum(x_i => Math.Cos(2 * Math.PI * x_i))) + 20 + Math.E;
        }

        // -100 <= xi <= 100
        // min at (0, 0, 0, ... , 0) and is 0
        private static double Schwefel1_2Function(double[] x)
        {
            double fitness = 0;

            for (int i = 0; i < x.Length; i++)
            {
                double innerSum = 0;

                for (int j = 0; j <= i; j++)
                {
                    innerSum += x[j];
                }

                fitness += Math.Pow(innerSum, 2);
            }

            return fitness;
        }

        // -10 <= xi <= 10
        // min at (0, 0, 0, ... , 0) and is 0
        private static double Alpine1Function(double[] x)
        {
            return x.Sum(x_i => Math.Abs(x_i * Math.Sin(x_i) + 0.1 * x_i));
        }

        // 0 <= xi <= 10
        // min at (7.917, 7.917, 7.917, ... , 7.917) and is 2.808^D
        private static double Alpine2Function(double[] x)
        {
            double product = 1;

            for(int i = 0; i < x.Length; i++)
            {
                product *= Math.Sqrt(x[i]) * Math.Sin(x[i]);
            }

            return -1 * product;
        }

        // -100 <= xi <= 100
        // min at (0, 0, 0, ... , 0) and is 0
        private static double Schwefel2_20Function(double[] x)
        {
            return x.Sum(Math.Abs);
        }

        // 0 <= xi <= 10
        // min at (0, 0, 0, ... , 0) and is 0
        private static double SphereFunction(double[] x)
        {
            return x.Sum(x_i => Math.Pow(x_i, 2));
        }

        // -100 <= xi <= 100
        // min at (0, 0, 0, ... , 0) and is 0
        private static double Step3Function(double[] x)
        {
            return x.Sum(x_i => Math.Floor(Math.Pow(x_i, 2)));
        }

        // -5 <= xi <= 10
        // min at (0, 0, 0, ... , 0) and is 0
        private static double ZakharovFunction(double[] x)
        {
            double sum1 = x.Sum(x_i => Math.Pow(x_i, 2));

            double sum2 = 0;

            for(int i = 0; i < x.Length; i++)
            {
                sum2 += (i + 1) * Math.Pow(x[i], 2);
            }

            return sum1 + 0.5 * Math.Pow(sum2, 2) + 0.5 * Math.Pow(sum2, 4);
        }

        // -5 <= xi <= 5
        // min at (0, 0, 0, ... , 0) and is 0
        private static double XinSheYangFunction(double[] x)
        {
            double sum = 0;

            Random random = new Random();

            for (int i = 0; i < x.Length; i++)
            {
                sum += random.NextDouble() * Math.Pow(Math.Abs(x[i]), (i + 1));
            }

            return sum;
        }

        // 0 <= xi <= pi
        // min at (0, 0, 0, ... , 0) and is 0
        private static double Trigonometric1Function(double[] x)
        {
            double sum = 0;

            for (int i = 0; i < x.Length; i++)
            {
                sum += Math.Pow(x.Length - x.Sum(Math.Cos) + (i + 1) * (1 - Math.Cos(x[i]) - Math.Sin(x[i])), 2);
            }

            return sum;
        }
    }
}
