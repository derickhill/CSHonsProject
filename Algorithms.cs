﻿using CSProject;
using MathNet.Numerics;
using MathNet.Numerics.Distributions;
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

        return fitnessFunc.getBest(population, populationSize);
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

        return fitnessFunc.getBest(population, populationSize);
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

        return fitnessFunc.getBest(population, populationSize);
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

    public static Pair[] convertToPairsArray(double[][] population)
    {
        Pair[] pairsPop = new Pair[population.Length];

        for(int i = 0; i < population.Length; i++)
        {
            pairsPop[i] = new Pair((double[])population[i].Clone());
        }

        return pairsPop;
    }

    public static Triple[] convertToTriplesArray(double[][] population)
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

    public static double[] RandomGA(double[][] pop, FitnessFunc fitnessFunc)
    {
        double[][] population = (double[][])pop.Clone();

        int dim = population[0].Length;

        double[] upperBound = fitnessFunc.getUpperBound(dim);
        double[] lowerBound = fitnessFunc.getLowerBound(dim);

        int iterations = 0;

        while (fitnessFunc.getCount() <= fitnessFunc.getMaxFunctionCalls())
        {
            iterations++;

            double[][] tempPopulation = new double[population.Length][];
            int tempPopulationSize = 0;

            tempPopulation[0] = fitnessFunc.getBest(population, population.Length);
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
        }

        return fitnessFunc.getFunctionCalls();
    }

    public static double[] GradientDescentGA(double[][] pop, FitnessFunc fitnessFunc, double learningRate)
    {
        double[][] population = (double[][])pop.Clone();

        int dim = population[0].Length;

        double[] upperBound = fitnessFunc.getUpperBound(dim);
        double[] lowerBound = fitnessFunc.getLowerBound(dim);

        int iterations = 0;

        while (fitnessFunc.getCount() <= fitnessFunc.getMaxFunctionCalls())
        {
            iterations++;

            double[][] tempPopulation = new double[population.Length][];
            int tempPopulationSize = 0;

            tempPopulation[0] = fitnessFunc.getBest(population, population.Length);
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

        }

        return fitnessFunc.getFunctionCalls();
    }

    public static double[] AdaGradGA(double[][] pop, FitnessFunc fitnessFunc, double initialLearningRate, double epsilon = 1e-8)
    {
        Pair[] population = convertToPairsArray(pop);

        int dim = population[0].First().Length;

        double[] upperBound = fitnessFunc.getUpperBound(dim);
        double[] lowerBound = fitnessFunc.getLowerBound(dim);

        int iterations = 0;

        while (fitnessFunc.getCount() <= fitnessFunc.getMaxFunctionCalls())
        {
            iterations++;

            Pair[] tempPopulation = new Pair[population.Length];
            int tempPopulationSize = 0;

            tempPopulation[0] = fitnessFunc.getBest(population, population.Length);
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

        }

        return fitnessFunc.getFunctionCalls();
    }

    public static double[] RMSPropGA(double[][] pop, FitnessFunc fitnessFunc, double initialLearningRate, double beta = 0.9, double epsilon = 1e-8)
    {
        Pair[] population = convertToPairsArray(pop);

        int dim = population[0].First().Length;

        double[] upperBound = fitnessFunc.getUpperBound(dim);
        double[] lowerBound = fitnessFunc.getLowerBound(dim);

        int iterations = 0;

        while (fitnessFunc.getCount() <= fitnessFunc.getMaxFunctionCalls())
        {
            iterations++;

            Pair[] tempPopulation = new Pair[population.Length];
            int tempPopulationSize = 0;

            tempPopulation[0] = fitnessFunc.getBest(population, population.Length);
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
                    squaredGradientSums[i] = beta * squaredGradientSums[i] + (1 - beta) * Math.Pow(gradient[i], 2);

                    individual[i] -= (initialLearningRate / Math.Sqrt(squaredGradientSums[i]) + epsilon) * gradient[i];

                    individual[i] = Math.Min(Math.Max(individual[i], lowerBound[i]), upperBound[i]);
                }

                mutated = new Pair(individual, squaredGradientSums);

                tempPopulation[tempPopulationSize] = mutated;
                tempPopulationSize++;
            }

            population = tempPopulation;
        }

        return fitnessFunc.getFunctionCalls();
    }

    public static double[] AdamGA(double[][] pop, FitnessFunc fitnessFunc, double initialLearningRate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
    {
        Triple[] population = convertToTriplesArray(pop);

        int dim = population[0].First().Length;

        double[] upperBound = fitnessFunc.getUpperBound(dim);
        double[] lowerBound = fitnessFunc.getLowerBound(dim);

        int iterations = 0;

        while (fitnessFunc.getCount() <= fitnessFunc.getMaxFunctionCalls())
        {
            iterations++;

            Triple[] tempPopulation = new Triple[population.Length];
            int tempPopulationSize = 0;

            tempPopulation[0] = fitnessFunc.getBest(population, population.Length);
            tempPopulationSize++;

            while (tempPopulationSize < population.Length)
            {
                Triple[] parents = select2(tempPopulation, tempPopulationSize, fitnessFunc);

                Triple crossedOver = coinFlipCrossover(parents[0], parents[1]);

                Triple mutated;

                double[] gradient = ApproximateGradient(fitnessFunc, crossedOver.First());

                double[] individual = crossedOver.First();
                double[] m = crossedOver.Second();
                double[] v = crossedOver.Last();

                for (int j = 0; j < dim; j++)
                {
                    // Update biased first moment estimate
                    m[j] = beta1 * m[j] + (1 - beta1) * gradient[j];

                    // Update biased second raw moment estimate
                    v[j] = beta2 * v[j] + (1 - beta2) * Math.Pow(gradient[j], 2);

                    // Bias-corrected first moment estimate
                    double mHat = m[j] / (1 - Math.Pow(beta1, iterations));

                    // Bias-corrected second raw moment estimate
                    double vHat = v[j] / (1 - Math.Pow(beta2, iterations));

                    // Update parameters
                    individual[j] -= initialLearningRate * mHat / (Math.Sqrt(vHat) + epsilon);

                    // Apply boundary constraints
                    individual[j] = Math.Min(Math.Max(individual[j], lowerBound[j]), upperBound[j]);
                }

                mutated = new Triple(individual, m, v);

                tempPopulation[tempPopulationSize] = mutated;
                tempPopulationSize++;
            }

            population = tempPopulation;
        }

        return fitnessFunc.getFunctionCalls();
    }
}
