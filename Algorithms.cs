using CSProject;
using MathNet.Numerics;
using MathNet.Numerics.Distributions;
using System;
using System.Security.Cryptography.X509Certificates;

public class Algorithms
{
    private static Random random = new Random(2315079);

    public static double epsilon = 1e-6;

    public static int method = 0;

    public static void setEps(double eps)
    {
        epsilon = eps;
    }

    public static void setMethod(int meth)
    {
        method = meth;
    }

    public static double[][] initialisePopulation(double[] lowerBound, double[] upperBound, int n)
    {
        int dimensions = lowerBound.Length;

        double[][] population = new double[n][];

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
        int tournamentSize = (int)Math.Ceiling(populationSize * 0.1);

        double[][] tournament = new double[tournamentSize][];

        int rnd = random.Next(0, populationSize);

        for (int i = 0; i < tournamentSize; i++)
        {
            tournament[i] = population[rnd];
        }

        return fitnessFunc.getBest(tournament, tournamentSize);
    }

    public static Pair tournamentSelection(Pair[] population, int populationSize, FitnessFunc fitnessFunc)
    {
        int tournamentSize = (int)Math.Ceiling(populationSize * 0.1);

        Pair[] tournament = new Pair[tournamentSize];

        int rnd = random.Next(0, populationSize);

        for (int i = 0; i < tournamentSize; i++)
        {
            tournament[i] = population[rnd];
        }

        return fitnessFunc.getBest(tournament, tournamentSize);
    }

    public static Triple tournamentSelection(Triple[] population, int populationSize, FitnessFunc fitnessFunc)
    {
        int tournamentSize = (int)Math.Ceiling(populationSize * 0.1);

        Triple[] tournament = new Triple[tournamentSize];

        int rnd = random.Next(0, populationSize);

        for (int i = 0; i < tournamentSize; i++)
        {
            tournament[i] = population[rnd];
        }

        return fitnessFunc.getBest(tournament, tournamentSize);
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

    // Approximate gradient
    public static double[] ApproximateGradient(FitnessFunc fitnessFunc, double[] x)
    {
        double[] gradient = new double[x.Length];

        // Centred differencing
        if (method == 0)
        {
            for (int i = 0; i < x.Length; i++)
            {
                double[] xPlusEps = (double[])x.Clone();
                double[] xMinusEps = (double[])x.Clone();

                xPlusEps[i] += epsilon;
                xMinusEps[i] -= epsilon;

                gradient[i] = (fitnessFunc.evaluate(xPlusEps) - fitnessFunc.evaluate(xMinusEps)) / (2 * epsilon);
            }
        }

        // Richardson's Extrapolation
        if(method == 1)
        {
            for (int i = 0; i < x.Length; i++)
            {
                double[] xPlusEps = (double[])x.Clone();
                double[] xMinusEps = (double[])x.Clone();

                double[] xPlus2Eps = (double[])x.Clone();
                double[] xMinus2Eps = (double[])x.Clone();

                xPlusEps[i] += epsilon;
                xMinusEps[i] -= epsilon;

                xPlus2Eps[i] += 2 * epsilon;
                xMinus2Eps[i] -= 2 * epsilon;

                gradient[i] = (-1 * fitnessFunc.evaluate(xPlus2Eps) + 8 * fitnessFunc.evaluate(xPlusEps) - 8 * fitnessFunc.evaluate(xMinusEps) + fitnessFunc.evaluate(xMinus2Eps)) / (12 * epsilon);
            }
        }

        // Forward differencing
        if (method == 2)
        {
            for (int i = 0; i < x.Length; i++)
            {
                double[] xPlusEps = (double[])x.Clone();

                xPlusEps[i] += epsilon;

                gradient[i] = (fitnessFunc.evaluate(xPlusEps) - fitnessFunc.evaluate(x)) / epsilon;
            }
        }

        // Backward differencing
        if (method == 3)
        {
            for (int i = 0; i < x.Length; i++)
            {
                double[] xMinusEps = (double[])x.Clone();

                xMinusEps[i] -= epsilon;

                gradient[i] = (fitnessFunc.evaluate(x) - fitnessFunc.evaluate(xMinusEps)) / epsilon;
            }
        }

        // Attempt at reducing function calls
        if (method == 4)
        {
            double[] xPlusEps = (double[])x.Clone();
            double[] xMinusEps = (double[])x.Clone();

            for (int i = 0; i < x.Length; i++)
            {
                xPlusEps[i] += epsilon;
                xMinusEps[i] -= epsilon;
            }

            for(int i = 0; i < x.Length; i++)
            {
                gradient[i] = (fitnessFunc.evaluate(xPlusEps) - fitnessFunc.evaluate(xMinusEps)) / (2 * epsilon);
            }
        }

        return gradient;
    }

    public static double[] RandomGA(double[][] pop, FitnessFunc fitnessFunc, double stepSize)
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
                double[][] parents = select2(population, population.Length, fitnessFunc);

                double[] crossedOver = coinFlipCrossover(parents[0], parents[1]);

                double[] mutated = (double[])crossedOver.Clone();

                for (int i = 0; i < dim; i++)
                {
                    MathNet.Numerics.Distributions.Normal normalDist = new(0, (upperBound[i] - lowerBound[i]) / stepSize);
                    double randomGaussianValue = normalDist.Sample();

                   // double mutation = lowerBound[i] + random.NextDouble() * (upperBound[i] - lowerBound[i]) * 1;

                    // Apply the mutation to the gene
                    mutated[i] += randomGaussianValue;

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

    public static double[] RandomGAnc(double[][] pop, FitnessFunc fitnessFunc, double stepSize)
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
                double[] parent = tournamentSelection(population, population.Length, fitnessFunc);

                double[] mutated = (double[])parent.Clone();

                for (int i = 0; i < dim; i++)
                {
                    MathNet.Numerics.Distributions.Normal normalDist = new(0, (upperBound[i] - lowerBound[i]) / stepSize);
                    double randomGaussianValue = normalDist.Sample();

                    // double mutation = lowerBound[i] + random.NextDouble() * (upperBound[i] - lowerBound[i]) * 1;

                    // Apply the mutation to the gene
                    mutated[i] += randomGaussianValue;

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

    public static double[] RandomGAnm(double[][] pop, FitnessFunc fitnessFunc, double stepSize)
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
                double[][] parents = select2(population, population.Length, fitnessFunc);

                double[] crossedOver = coinFlipCrossover(parents[0], parents[1]);

                double[] mutated = (double[])crossedOver.Clone();

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
                double[][] parents = select2(population, population.Length, fitnessFunc);

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

    public static double[] GradientDescentGAnc(double[][] pop, FitnessFunc fitnessFunc, double learningRate)
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
                double[] parent = tournamentSelection(population, population.Length, fitnessFunc);

                double[] mutated = (double[])parent.Clone();

                double[] gradient = ApproximateGradient(fitnessFunc, parent);

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

    
    public static double[] AdaGradGA(double[][] pop, FitnessFunc fitnessFunc, double initialLearningRate)
    {
        Pair[] population = convertToPairsArray(pop);

        double eps = 1e-8;

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
                Pair[] parents = select2(population, population.Length, fitnessFunc);

                Pair crossedOver = coinFlipCrossover(parents[0], parents[1]);

                Pair mutated;

                double[] gradient = ApproximateGradient(fitnessFunc, crossedOver.First());

                double[] individual = crossedOver.First();
                double[] squaredGradientSums = crossedOver.Last();

                for (int i = 0; i < dim; i++)
                {
                    squaredGradientSums[i] += Math.Pow(gradient[i], 2);

                    // Compute the adaptive learning rate
                    double adaptiveLearningRate = initialLearningRate / Math.Sqrt(squaredGradientSums[i] + eps);

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

            Console.WriteLine("Iteration: " + iterations + "\t\t\t\tFunction calls: " + fitnessFunc.count);

        }

        return fitnessFunc.getFunctionCalls();
    }

    public static double[] AdaGradGAnc(double[][] pop, FitnessFunc fitnessFunc, double initialLearningRate)
    {
        Pair[] population = convertToPairsArray(pop);

        double eps = 1e-8;

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
                Pair parent = tournamentSelection(population, population.Length, fitnessFunc);

                Pair mutated;

                double[] gradient = ApproximateGradient(fitnessFunc, parent.First());

                double[] individual = parent.First();
                double[] squaredGradientSums = parent.Last();

                for (int i = 0; i < dim; i++)
                {
                    squaredGradientSums[i] += Math.Pow(gradient[i], 2);

                    // Compute the adaptive learning rate
                    double adaptiveLearningRate = initialLearningRate / Math.Sqrt(squaredGradientSums[i] + eps);

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

            Console.WriteLine("Iteration: " + iterations + "\t\t\t\tFunction calls: " + fitnessFunc.count);

        }

        return fitnessFunc.getFunctionCalls();
    }


    public static double[] AdaGrad(double[][] pop, FitnessFunc fitnessFunc, double initialLearningRate)
    {
        double eps = 1e-8;

        int dim = pop[0].Length;

        Pair[] population = convertToPairsArray(pop);

        double[] upperBound = fitnessFunc.getUpperBound(dim);
        double[] lowerBound = fitnessFunc.getLowerBound(dim);

        int iterations = 0;

        double[] individual = pop[random.Next(pop.Length)];
        double[] squaredGradientSums = new double[dim];

        while (fitnessFunc.getCount() <= fitnessFunc.getMaxFunctionCalls())
        {
            iterations++;

            for (int k = 0; k < pop.Length; k++)
            {

                individual = population[k].First();
                squaredGradientSums = population[k].Last();

                double[] gradient = ApproximateGradient(fitnessFunc, individual);

                for (int i = 0; i < dim; i++)
                {
                    squaredGradientSums[i] += Math.Pow(gradient[i], 2);

                    // Compute the adaptive learning rate
                    double adaptiveLearningRate = initialLearningRate / Math.Sqrt(squaredGradientSums[i] + eps);

                    // Update the parameter
                    individual[i] -= adaptiveLearningRate * gradient[i];

                    // Apply boundary constraints
                    individual[i] = Math.Min(Math.Max(individual[i], lowerBound[i]), upperBound[i]);
                }

                population[k] = new Pair(individual, squaredGradientSums);
            }
        }

        return fitnessFunc.getFunctionCalls();
    }

    

    public static double[] RMSPropGA(double[][] pop, FitnessFunc fitnessFunc, double initialLearningRate, double beta = 0.9)
    {
        Pair[] population = convertToPairsArray(pop);

        double eps = 1e-8;

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
                Pair[] parents = select2(population, population.Length, fitnessFunc);

                Pair crossedOver = coinFlipCrossover(parents[0], parents[1]);

                Pair mutated;

                double[] gradient = ApproximateGradient(fitnessFunc, crossedOver.First());

                double[] individual = crossedOver.First();
                double[] squaredGradientSums = crossedOver.Last();

                for (int i = 0; i < dim; i++)
                {
                    squaredGradientSums[i] = beta * squaredGradientSums[i] + (1 - beta) * Math.Pow(gradient[i], 2);

                    individual[i] -= (initialLearningRate / Math.Sqrt(squaredGradientSums[i]) + eps) * gradient[i];

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

    public static double[] RMSPropGAnc(double[][] pop, FitnessFunc fitnessFunc, double initialLearningRate, double beta = 0.9)
    {
        Pair[] population = convertToPairsArray(pop);

        double eps = 1e-8;

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
                Pair parent = tournamentSelection(population, population.Length, fitnessFunc);

                Pair mutated;

                double[] gradient = ApproximateGradient(fitnessFunc, parent.First());

                double[] individual = parent.First();
                double[] squaredGradientSums = parent.Last();

                for (int i = 0; i < dim; i++)
                {
                    squaredGradientSums[i] = beta * squaredGradientSums[i] + (1 - beta) * Math.Pow(gradient[i], 2);

                    individual[i] -= (initialLearningRate / Math.Sqrt(squaredGradientSums[i]) + eps) * gradient[i];

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

    public static double[] RMSProp(double[][] pop, FitnessFunc fitnessFunc, double initialLearningRate, double beta = 0.9)
    {
        double eps = 1e-8;

        int dim = pop[0].Length;

        Pair[] population = convertToPairsArray(pop);

        double[] upperBound = fitnessFunc.getUpperBound(dim);
        double[] lowerBound = fitnessFunc.getLowerBound(dim);

        int iterations = 0;

        double[] individual = pop[random.Next(pop.Length)];
        double[] squaredGradientSums = new double[dim];

        while (fitnessFunc.getCount() <= fitnessFunc.getMaxFunctionCalls())
        {
            iterations++;

            for (int k = 0; k < pop.Length; k++)
            {

                individual = population[k].First();
                squaredGradientSums = population[k].Last();

                double[] gradient = ApproximateGradient(fitnessFunc, individual);

                for (int i = 0; i < dim; i++)
                {
                    squaredGradientSums[i] = beta * squaredGradientSums[i] + (1 - beta) * Math.Pow(gradient[i], 2);

                    individual[i] -= (initialLearningRate / Math.Sqrt(squaredGradientSums[i]) + eps) * gradient[i];

                    individual[i] = Math.Min(Math.Max(individual[i], lowerBound[i]), upperBound[i]);
                }

                population[k] = new Pair(individual, squaredGradientSums);
            }
        }

        return fitnessFunc.getFunctionCalls();
    }

    public static double[] AdaDeltaGA(double[][] pop, FitnessFunc fitnessFunc, double rho = 0.99)
    {
        Triple[] population = convertToTriplesArray(pop);

        double eps = 1e-6;

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
                Triple[] parents = select2(population, population.Length, fitnessFunc);

                Triple crossedOver = coinFlipCrossover(parents[0], parents[1]);

                Triple mutated;

                double[] gradient = ApproximateGradient(fitnessFunc, crossedOver.First());

                double[] individual = crossedOver.First();
                double[] gradientsSquaredSums = crossedOver.Second();
                double[] parameterUpdatesSquared = crossedOver.Last();

                for (int j = 0; j < dim; j++)
                {
                    gradientsSquaredSums[j] = rho * gradientsSquaredSums[j] + (1 - rho) * Math.Pow(gradient[j], 2);

                    double delta = -Math.Sqrt((parameterUpdatesSquared[j] + eps) / (gradientsSquaredSums[j] + eps)) * gradient[j];

                    individual[j] += delta;

                    parameterUpdatesSquared[j] = rho * parameterUpdatesSquared[j] + (1 - rho) * Math.Pow(delta, 2);

                    // Apply boundary constraints
                    individual[j] = Math.Min(Math.Max(individual[j], lowerBound[j]), upperBound[j]);
                }

                mutated = new Triple(individual, gradientsSquaredSums, parameterUpdatesSquared);

                tempPopulation[tempPopulationSize] = mutated;
                tempPopulationSize++;
            }

            population = tempPopulation;
        }

        return fitnessFunc.getFunctionCalls();
    }

    public static double[] AdaDelta(double[][] pop, FitnessFunc fitnessFunc, double rho = 0.99)
    {
        double eps = 1e-6;

        int dim = pop[0].Length;

        Triple[] population = convertToTriplesArray(pop);

        double[] upperBound = fitnessFunc.getUpperBound(dim);
        double[] lowerBound = fitnessFunc.getLowerBound(dim);

        int iterations = 0;

        double[] individual = pop[random.Next(pop.Length)];
        double[] gradientsSquaredSums = new double[dim];
        double[] parameterUpdatesSquared = new double[dim];

        while (fitnessFunc.getCount() <= fitnessFunc.getMaxFunctionCalls())
        {
            iterations++;

            for (int k = 0; k < pop.Length; k++)
            {
                individual = population[k].First();
                gradientsSquaredSums = population[k].Second();
                parameterUpdatesSquared = population[k].Last();

                double[] gradient = ApproximateGradient(fitnessFunc, individual);

                for (int j = 0; j < dim; j++)
                {
                    gradientsSquaredSums[j] = rho * gradientsSquaredSums[j] + (1 - rho) * Math.Pow(gradient[j], 2);

                    double delta = -Math.Sqrt((parameterUpdatesSquared[j] + eps) / (gradientsSquaredSums[j] + eps)) * gradient[j];

                    individual[j] += delta;

                    parameterUpdatesSquared[j] = rho * parameterUpdatesSquared[j] + (1 - rho) * Math.Pow(delta, 2);

                    // Apply boundary constraints
                    individual[j] = Math.Min(Math.Max(individual[j], lowerBound[j]), upperBound[j]);
                }

                population[k] = new Triple(individual, gradientsSquaredSums, parameterUpdatesSquared);
            }
        }

        return fitnessFunc.getFunctionCalls();
    }

    public static double[] AdaDeltaGAnc(double[][] pop, FitnessFunc fitnessFunc, double rho = 0.99)
    {
        Triple[] population = convertToTriplesArray(pop);

        double eps = 1e-6;

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
                Triple parent = tournamentSelection(population, population.Length, fitnessFunc);

                Triple mutated;

                double[] gradient = ApproximateGradient(fitnessFunc, parent.First());

                double[] individual = parent.First();
                double[] gradientsSquaredSums = parent.Second();
                double[] parameterUpdatesSquared = parent.Last();

                for (int j = 0; j < dim; j++)
                {
                    gradientsSquaredSums[j] = rho * gradientsSquaredSums[j] + (1 - rho) * Math.Pow(gradient[j], 2);

                    double delta = -Math.Sqrt((parameterUpdatesSquared[j] + eps) / (gradientsSquaredSums[j] + eps)) * gradient[j];

                    individual[j] += delta;

                    parameterUpdatesSquared[j] = rho * parameterUpdatesSquared[j] + (1 - rho) * Math.Pow(delta, 2);

                    // Apply boundary constraints
                    individual[j] = Math.Min(Math.Max(individual[j], lowerBound[j]), upperBound[j]);
                }

                mutated = new Triple(individual, gradientsSquaredSums, parameterUpdatesSquared);

                tempPopulation[tempPopulationSize] = mutated;
                tempPopulationSize++;
            }

            population = tempPopulation;
        }

        return fitnessFunc.getFunctionCalls();
    }

    public static double[] AdamGA(double[][] pop, FitnessFunc fitnessFunc, double initialLearningRate, double beta1 = 0.9, double beta2 = 0.999)
    {
        Triple[] population = convertToTriplesArray(pop);

        double eps = 1e-8;

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
                Triple[] parents = select2(population, population.Length, fitnessFunc);

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
                    individual[j] -= initialLearningRate * mHat / (Math.Sqrt(vHat) + eps);

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

    public static double[] AdamGAnc(double[][] pop, FitnessFunc fitnessFunc, double initialLearningRate, double beta1 = 0.9, double beta2 = 0.999)
    {
        Triple[] population = convertToTriplesArray(pop);

        double eps = 1e-8;

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
                Triple parent = tournamentSelection(population, population.Length, fitnessFunc);

                Triple crossedOver = parent;

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
                    individual[j] -= initialLearningRate * mHat / (Math.Sqrt(vHat) + eps);

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

    public static double[] Adam(double[][] pop, FitnessFunc fitnessFunc, double initialLearningRate, double beta1 = 0.9, double beta2 = 0.999)
    {
        double eps = 1e-8;

        int dim = pop[0].Length;

        Triple[] population = convertToTriplesArray(pop);

        double[] upperBound = fitnessFunc.getUpperBound(dim);
        double[] lowerBound = fitnessFunc.getLowerBound(dim);

        int iterations = 0;

        double[] individual = pop[random.Next(pop.Length)];
        double[] m = new double[dim];
        double[] v = new double[dim];

        while (fitnessFunc.getCount() <= fitnessFunc.getMaxFunctionCalls())
        {
            iterations++;

            for (int k = 0; k < pop.Length; k++)
            {
                individual = population[k].First();
                m = population[k].Second();
                v = population[k].Last();

                double[] gradient = ApproximateGradient(fitnessFunc, individual);

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
                    individual[j] -= initialLearningRate * mHat / (Math.Sqrt(vHat) + eps);

                    // Apply boundary constraints
                    individual[j] = Math.Min(Math.Max(individual[j], lowerBound[j]), upperBound[j]);
                }

                population[k] = new Triple(individual, m, v);
            }
        }

        return fitnessFunc.getFunctionCalls();
    }

    public static double[] ImprovedAdamGA(double[][] pop, FitnessFunc fitnessFunc, double initialLearningRate, double beta1 = 0.9, double beta2 = 0.999)
    {
        Triple[] population = convertToTriplesArray(pop);

        double eps = 1e-8;

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
                Triple[] parents = select2(population, population.Length, fitnessFunc);

                Triple crossedOver = coinFlipCrossover(parents[0], parents[1]);

                Triple mutated;

                double[] gradient = ApproximateGradient(fitnessFunc, crossedOver.First());

                double[] individual = crossedOver.First();
                double[] m = crossedOver.Second();
                double[] v = crossedOver.Last();

                for(int i = 0; i < dim; i++)
                {
                    individual[i] = individual[i] / Math.Abs(individual[i]);
                    gradient[i] = gradient[i] - gradient[i] * individual[i] * individual[i];
                    m[i] = beta2 * m[i] + (1 - beta1) * gradient[i];
                    v[i] = beta2 * v[i] + (1 - beta2) * Math.Pow(gradient[i], 2);
                    double mHat = m[i] / (1 - Math.Pow(beta1, iterations));
                    double vHat = v[i] / (1 - Math.Pow(beta2, iterations));
                    individual[i] = individual[i] - initialLearningRate * mHat / (Math.Sqrt(vHat) + eps);
                    individual[i] = individual[i] / Math.Abs(individual[i]);

                }

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
                    individual[j] -= initialLearningRate * mHat / (Math.Sqrt(vHat) + eps);

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
