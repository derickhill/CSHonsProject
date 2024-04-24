using CSProject;
using System;
using System.Diagnostics;

public class Adam
{
    public static KeyValuePair<double[], double[][]> AdamMutation(int iteration, KeyValuePair<double[], double[][]> offspring, double[] lowerBound, double[] upperBound, Func<double[], double> fitnessFunc, bool minimisation = true)
    {
        KeyValuePair<double[], double[][]>  mutated = AdamOptimiser(iteration, fitnessFunc, offspring, 1.9, lowerBound, upperBound);

        return mutated;
    }

    // Adam optimiser for adaptive learning rate
    public static KeyValuePair<double[], double[][]> AdamOptimiser(int iteration, Func<double[], double> fitnessFunc, KeyValuePair<double[], double[][]> offspring, double learningRate, double[] lowerBound, double[] upperBound, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
    {
        int dimensions = offspring.Key.Length;

        double[] currentSolution = (double[])offspring.Key.Clone();
        // First moment estimate:
        double[] m = (double[])offspring.Value[0].Clone();
        // Second moment estimate:
        double[] v = (double[])offspring.Value[1].Clone();

        double[] gradient = Project.ApproximateGradient(fitnessFunc, currentSolution);

            for (int j = 0; j < dimensions; j++)
            {
                // Update biased first moment estimate
                m[j] = beta1 * m[j] + (1 - beta1) * gradient[j];

                // Update biased second raw moment estimate
                v[j] = beta2 * v[j] + (1 - beta2) * Math.Pow(gradient[j], 2);

                // Bias-corrected first moment estimate
                double mHat = m[j] / (1 - Math.Pow(beta1, iteration));

                // Bias-corrected second raw moment estimate
                double vHat = v[j] / (1 - Math.Pow(beta2, iteration));

                // Update parameters
                currentSolution[j] -= 0.6 * mHat / (Math.Sqrt(vHat) + epsilon);

                // Apply boundary constraints
                currentSolution[j] = Math.Min(Math.Max(currentSolution[j], lowerBound[j]), upperBound[j]);
            }
 
        return new KeyValuePair<double[], double[][]>(currentSolution, [m, v]);
    }

    public static KeyValuePair<double[], double[][]>[] initialisePopulation(double[] lowerBound, double[] upperBound, int n)
    {
        int dimensions = lowerBound.Length;

        KeyValuePair<double[], double[][]>[] populationWithLearningRates = new KeyValuePair<double[], double[][]>[n];

        Random random = new Random();

        double[] m = new double[dimensions];
        Array.Fill(m, 0);

        double[] v = new double[dimensions];
        Array.Fill(v, 0);

        for (int i = 0; i < n; i++)
        {

            double[] curIndividual = new double[dimensions];

            for (int j = 0; j < dimensions; j++)
            {
                curIndividual[j] = random.NextDouble() * (upperBound[j] - lowerBound[j]) + lowerBound[j];
            }

            populationWithLearningRates[i] = new KeyValuePair<double[], double[][]>(curIndividual, [m, v]);
        }

        return populationWithLearningRates;
    }

    public static KeyValuePair<double[], double[][]> coinFlipCrossover(KeyValuePair<double[], double[][]> parent1, KeyValuePair<double[], double[][]> parent2)
    {
        Random random = new Random();

        int dimensions = parent1.Key.Length;

        double[] child = new double[dimensions];
        double[] m = new double[dimensions];
        double[] v = new double[dimensions];

        for (int i = 0; i < dimensions; i++)
        {
            double rnd = random.Next(0, 2);

            if (rnd == 0)
            {
                child[i] = parent1.Key[i];
                m[i] = parent1.Value[0][i];
                v[i] = parent1.Value[1][i];
            }
            else
            {
                child[i] = parent2.Key[i];
                m[i] = parent2.Value[0][i];
                v[i] = parent2.Value[1][i];
            }
        }

        return new KeyValuePair<double[], double[][]>(child, [m, v]);
    }

    public static KeyValuePair<double[], double[][]> tournamentSelection(KeyValuePair<double[], double[][]>[] population, int populationSize, Func<double[], double> fitnessFunc, bool minimisation = true)
    {
        int tournamentSize = (int)Math.Ceiling(populationSize * 0.8);

        KeyValuePair<double[], double[][]>[] tournament = new KeyValuePair<double[], double[][]>[tournamentSize];

        Random random = new Random();

        int rnd = random.Next(0, populationSize);

        for (int i = 0; i < tournamentSize; i++)
        {
            tournament[i] = population[rnd];
        }

        Array.Sort(tournament, (x, y) =>
        {
            if (minimisation)
                return fitnessFunc(x.Key).CompareTo(fitnessFunc(y.Key));
            else
                return fitnessFunc(y.Key).CompareTo(fitnessFunc(x.Key));
        });

        return tournament[0];
    }

    public static KeyValuePair<double[], double[][]>[] select2(KeyValuePair<double[], double[][]>[] population, int populationSize, Func<double[], double> fitnessFunc, bool minimisation = true)
    {
        return [tournamentSelection(population, populationSize, fitnessFunc, minimisation), tournamentSelection(population, populationSize, fitnessFunc, minimisation)];
    }

    public static double[] GA(double[] ga, KeyValuePair<double[], double[][]>[] population, Func<double[], double> fitnessFunc, double[] lowerBound, double[] upperBound, Func<int, KeyValuePair<double[], double[][]>, double[], double[], Func<double[], double>, bool, KeyValuePair<double[], double[][]>> mutationFunc, bool minimisation = true, double error = 0.01, int maxIterations = int.MaxValue)
    {
        //double[][] population = initialisePopulation(lowerBound, upperBound, populationSize);
        var stopwatch = new Stopwatch();
        stopwatch.Start();

        Array.Sort(population, (x, y) =>
        {
            if (minimisation)
                return fitnessFunc(x.Key).CompareTo(fitnessFunc(y.Key));
            else
                return fitnessFunc(y.Key).CompareTo(fitnessFunc(x.Key));
        });

        int iterations = 0;

        while (iterations < 100)
        {
            ga[iterations] += fitnessFunc(population[0].Key);

            iterations++;

            KeyValuePair<double[], double[][]>[] tempPopulation = new KeyValuePair<double[], double[][]>[population.Length];
            int tempPopulationSize = 0;

            tempPopulation[0] = population[0];
            tempPopulationSize++;

            while (tempPopulationSize < population.Length)
            {
                KeyValuePair<double[], double[][]>[] parents = select2(tempPopulation, tempPopulationSize, fitnessFunc, minimisation);

                KeyValuePair<double[], double[][]> crossedOver = coinFlipCrossover(parents[0], parents[1]);

                KeyValuePair<double[], double[][]> mutated = mutationFunc( iterations, crossedOver, lowerBound, upperBound, fitnessFunc, minimisation);

                tempPopulation[tempPopulationSize] = mutated;
                tempPopulationSize++;
            }

            population = tempPopulation;

            Array.Sort(population, (x, y) =>
            {
                if (minimisation)
                    return fitnessFunc(x.Key).CompareTo(fitnessFunc(y.Key));
                else
                    return fitnessFunc(y.Key).CompareTo(fitnessFunc(x.Key));
            });

            // Console.WriteLine("Fitness of fittest individual: " + fitnessFunc(population[0].Key));
            //Console.WriteLine("Learning rate: [" + (0.01/(population[0].Value[1][0] * population[0].Value[1][0])));
        }

        stopwatch.Stop();

        Console.WriteLine("Iterations: " + iterations);
        Console.WriteLine("Fitness of fittest individual: " + fitnessFunc(population[0].Key));
        Console.WriteLine("Fittest individual: [" + string.Join(", ", population[0].Key) + "]");

        return [iterations, fitnessFunc(population[0].Key), stopwatch.Elapsed.TotalMilliseconds];
    }
}
