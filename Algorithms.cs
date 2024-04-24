using CSProject;
using System;

public class Algorithms
{
    public static double[][] initialisePopulation(double[] lowerBound, double[] upperBound, int n)
    {
        int dimensions = lowerBound.Length;

        double[][] population = new double[n][];

        Random random = new Random();

        for (int i = 0; i < n; i++)
        {
            double[] curIndividual = new double[dimensions];

            for (int j = 0; j < dimensions; j++)
            {
                curIndividual[j] = random.NextDouble() * (upperBound[j] - lowerBound[j]) + lowerBound[j];
            }

            population[i] = curIndividual;
        }

        return population;
    }

    public void RandomGA(double[][] pop, FitnessFunc fitnessFunc)
    {
        double[][] population = (double[][])pop.Clone();

        Array.Sort(population, (x, y) =>
        {
            return fitnessFunc.evaluate(x).CompareTo(fitnessFunc.evaluate(y));
        });

        int iterations = 0;

        while (fitnessFunc.getCount() <= 50000)
        {
            iterations++;

            double[][] tempPopulation = new double[population.Length][];
            int tempPopulationSize = 0;

            tempPopulation[0] = population[0];
            tempPopulationSize++;

            while (tempPopulationSize < population.Length)
            {
                double[][] parents = select2(tempPopulation, tempPopulationSize, fitnessFunc, minimisation);

                double[] crossedOver = coinFlipCrossover(parents[0], parents[1], fitnessFunc);

                double[] mutated = mutationFunc(crossedOver, lowerBound, upperBound, fitnessFunc, minimisation);

                tempPopulation[tempPopulationSize] = mutated;
                tempPopulationSize++;
            }

            population = tempPopulation;

            Array.Sort(population, (x, y) =>
            {
                return fitnessFunc.evaluate(x).CompareTo(fitnessFunc.evaluate(y));
            });

        }
    }

    public void GradientDescentGA(double[][] population, FitnessFunc fitnessFunc)
    {

    }

    public void AdaGradGA(double[][] population, FitnessFunc fitnessFunc)
    {

    }

    public void RMSPropGA(double[][] population, FitnessFunc fitnessFunc)
    {

    }

    public void AdamGA(double[][] population, FitnessFunc fitnessFunc)
    {

    }
}
