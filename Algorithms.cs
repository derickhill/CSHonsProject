using CSProject;
using MathNet.Numerics;
using System;
using System.Security.Cryptography.X509Certificates;

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

    public static double[] tournamentSelection(double[][] population, int populationSize, FitnessFunc fitnessFunc)
    {
        int tournamentSize = (int)Math.Ceiling(populationSize * 0.8);

        double[][] tournament = new double[tournamentSize][];

        Random random = new Random();

        int rnd = random.Next(0, populationSize);

        for (int i = 0; i < tournamentSize; i++)
        {
            tournament[i] = population[rnd];
        }

        Array.Sort(tournament, (x, y) =>
        {
            return fitnessFunc.evaluate(x).CompareTo(fitnessFunc.evaluate(y));
        });

        return tournament[0];
    }

    public static Pair tournamentSelection(Pair[] population, int populationSize, FitnessFunc fitnessFunc)
    {
        int tournamentSize = (int)Math.Ceiling(populationSize * 0.8);

        Pair[] tournament = new Pair[tournamentSize];

        Random random = new Random();

        int rnd = random.Next(0, populationSize);

        for (int i = 0; i < tournamentSize; i++)
        {
            tournament[i] = population[rnd];
        }

        Array.Sort(tournament, (x, y) =>
        {
            return fitnessFunc.evaluate(x.First()).CompareTo(fitnessFunc.evaluate(y.First()));
        });

        return tournament[0];
    }

    public static Triple tournamentSelection(Triple[] population, int populationSize, FitnessFunc fitnessFunc)
    {
        int tournamentSize = (int)Math.Ceiling(populationSize * 0.8);

        Triple[] tournament = new Triple[tournamentSize];

        Random random = new Random();

        int rnd = random.Next(0, populationSize);

        for (int i = 0; i < tournamentSize; i++)
        {
            tournament[i] = population[rnd];
        }

        Array.Sort(tournament, (x, y) =>
        {
            return fitnessFunc.evaluate(x.First()).CompareTo(fitnessFunc.evaluate(y.First()));
        });

        return tournament[0];
    }

    public static double[][] select2(double[][] population, int populationSize, FitnessFunc fitnessFunc)
    {
        return [tournamentSelection(population, populationSize, fitnessFunc), tournamentSelection(population, populationSize, fitnessFunc)];
    }

    public static Pair[] select2(Pair[] population, int populationSize, FitnessFunc fitnessFunc)
    {
        return [tournamentSelection(population, populationSize, fitnessFunc), tournamentSelection(population, populationSize, fitnessFunc)];
    }

    public static Triple[] select2(Triple[] population, int populationSize, FitnessFunc fitnessFunc)
    {
        return [tournamentSelection(population, populationSize, fitnessFunc), tournamentSelection(population, populationSize, fitnessFunc)];
    }

    public static double[] coinFlipCrossover(double[] parent1, double[] parent2)
    {
        Random random = new Random();

        double[] child = new double[parent1.Length];

        for (int i = 0; i < parent1.Length; i++)
        {
            double rnd = random.Next(0, 2);

            if (rnd == 0) child[i] = parent1[i];
            else child[i] = parent2[i];
        }

        return child;
    }

    public static Pair coinFlipCrossover(Pair parent1, Pair parent2)
    {
        Random random = new Random();

        double[] child = new double[parent1.First().Length];
        double[] gradients = new double[parent1.First().Length];

        for (int i = 0; i < parent1.First().Length; i++)
        {
            double rnd = random.Next(0, 2);

            if (rnd == 0)
            {
                child[i] = parent1.First()[i];
                gradients[i] = parent1.Last()[i];
            }
            else
            {
                child[i] = parent2.First()[i];
                gradients[i] = parent2.Last()[i];
            }
        }

        return new Pair(child, gradients);
    }

    public static Triple coinFlipCrossover(Triple parent1, Triple parent2)
    {
        Random random = new Random();

        double[] child = new double[parent1.First().Length];
        double[] m = new double[parent1.First().Length];
        double[] v = new double[parent1.First().Length];

        for (int i = 0; i < parent1.First().Length; i++)
        {
            double rnd = random.Next(0, 2);

            if (rnd == 0)
            {
                child[i] = parent1.First()[i];
                m[i] = parent1.Second()[i];
                v[i] = parent1.Last()[i];
            }
            else
            {
                child[i] = parent2.First()[i];
                m[i] = parent2.Second()[i];
                v[i] = parent2.Last()[i];
            }
        }

        return new Triple(child, m, v);
    }

    public Pair[] convertToPairsArray(double[][] population)
    {
        Pair[] pairsPop = new Pair[population.Length];

        for(int i = 0; i < population.Length; i++)
        {
            pairsPop[i] = new Pair((double[])population[i].Clone());
        }

        return pairsPop;
    }

    public Triple[] convertToTriplesArray(double[][] population)
    {
        Triple[] triplesPop = new Triple[population.Length];

        for (int i = 0; i < population.Length; i++)
        {
            triplesPop[i] = new Triple((double[])population[i].Clone());
        }

        return triplesPop;
    }

    // Approximate gradient using finite differences
    public static double[] ApproximateGradient(FitnessFunc fitnessFunc, double[] x, double epsilon = 1e-6)
    {
        double[] gradient = new double[x.Length];

        for (int i = 0; i < x.Length; i++)
        {
            double[] xPlusEps = (double[])x.Clone();
            double[] xMinusEps = (double[])x.Clone();

            xPlusEps[i] += epsilon;
            xMinusEps[i] -= epsilon;

            gradient[i] = (fitnessFunc.evaluate(xPlusEps) - fitnessFunc.evaluate(xMinusEps)) / (2 * epsilon);
        }

        return gradient;
    }

    public void RandomGA(double[][] pop, FitnessFunc fitnessFunc)
    {
        double[][] population = (double[][])pop.Clone();

        int dim = population[0].Length;

        double[] upperBound = fitnessFunc.getUpperBound(dim);
        double[] lowerBound = fitnessFunc.getLowerBound(dim);

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
                double[][] parents = select2(tempPopulation, tempPopulationSize, fitnessFunc);

                double[] crossedOver = coinFlipCrossover(parents[0], parents[1]);

                double[] mutated = (double[])crossedOver.Clone();

                Random random = new Random();

                for (int i = 0; i < dim; i++)
                {
                    double mutation = lowerBound[i] + random.NextDouble() * (upperBound[i] - lowerBound[i]) * 1;

                    // Apply the mutation to the gene
                    mutated[i] += mutation;

                    // Apply boundary constraints
                    mutated[i] = Math.Min(Math.Max(mutated[i], lowerBound[i]), upperBound[i]);
                }

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

    public void GradientDescentGA(double[][] pop, FitnessFunc fitnessFunc, double learningRate)
    {
        double[][] population = (double[][])pop.Clone();

        int dim = population[0].Length;

        double[] upperBound = fitnessFunc.getUpperBound(dim);
        double[] lowerBound = fitnessFunc.getLowerBound(dim);

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
                double[][] parents = select2(tempPopulation, tempPopulationSize, fitnessFunc);

                double[] crossedOver = coinFlipCrossover(parents[0], parents[1]);

                double[] mutated = (double[])crossedOver.Clone();

                double[] gradient = ApproximateGradient(fitnessFunc, crossedOver);

                for (int i = 0; i < dim; i++)
                {
                    mutated[i] -= learningRate * gradient[i];

                    // Apply boundary constraints
                    mutated[i] = Math.Min(Math.Max(mutated[i], lowerBound[i]), upperBound[i]);
                }

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

    public void AdaGradGA(double[][] pop, FitnessFunc fitnessFunc, double initialLearningRate, double epsilon = 1e-8)
    {
        Pair[] population = convertToPairsArray(pop);

        int dim = population[0].First().Length;

        double[] upperBound = fitnessFunc.getUpperBound(dim);
        double[] lowerBound = fitnessFunc.getLowerBound(dim);

        Array.Sort(population, (x, y) =>
        {
            return fitnessFunc.evaluate(x.First()).CompareTo(fitnessFunc.evaluate(y.First()));
        });

        int iterations = 0;

        while (fitnessFunc.getCount() <= 50000)
        {
            iterations++;

            Pair[] tempPopulation = new Pair[population.Length];
            int tempPopulationSize = 0;

            tempPopulation[0] = population[0];
            tempPopulationSize++;

            while (tempPopulationSize < population.Length)
            {
                Pair[] parents = select2(tempPopulation, tempPopulationSize, fitnessFunc);

                Pair crossedOver = coinFlipCrossover(parents[0], parents[1]);

                Pair mutated;

                double[] gradient = ApproximateGradient(fitnessFunc, crossedOver.First());

                double[] individual = crossedOver.First();
                double[] squaredGradientSums = crossedOver.Last();

                for (int i = 0; i < dim; i++)
                {
                    squaredGradientSums[i] += Math.Pow(gradient[i], 2);

                    // Compute the adaptive learning rate
                    double adaptiveLearningRate = initialLearningRate / Math.Sqrt(squaredGradientSums[i] + epsilon);

                    // Update the parameter
                    individual[i] -= adaptiveLearningRate * gradient[i];

                    // Apply boundary constraints
                    individual[i] = Math.Min(Math.Max(individual[i], lowerBound[i]), upperBound[i]);
                }

                mutated = new Pair(individual, squaredGradientSums);

                tempPopulation[tempPopulationSize] = mutated;
                tempPopulationSize++;
            }

            population = tempPopulation;

            Array.Sort(population, (x, y) =>
            {
                return fitnessFunc.evaluate(x.First()).CompareTo(fitnessFunc.evaluate(y.First()));
            });

        }
    }

    public void RMSPropGA(double[][] population, FitnessFunc fitnessFunc, double initialLearningRate)
    {

    }

    public void AdamGA(double[][] population, FitnessFunc fitnessFunc, double initialLearningRate)
    {

    }
}
