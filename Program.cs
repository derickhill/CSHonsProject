using CSProject;
using MathNet.Numerics;
using MathNet.Numerics.Distributions;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

public class Project
{
    static double[] ga1 = new double[100];
    static double[] ga2 = new double[100];
    static double[] ga3 = new double[100];
    static double[] ga4 = new double[100];
    static double[] ga5 = new double[100];
    static double[] ga6 = new double[300];

    static double learningRate = 0.1;
    public static double[][] initialisePopulation(double[] lowerBound, double[] upperBound, int n)
    {
        int dimensions = lowerBound.Length;

        double[][] population = new double[n][];

        Random random = new Random();

        for(int i = 0; i < n; i++) 
        {
            double[] curIndividual = new double[dimensions];

            for(int j = 0; j < dimensions; j++)
            {
                curIndividual[j] = random.NextDouble() * (upperBound[j] - lowerBound[j]) + lowerBound[j];
            }

            population[i] = curIndividual;
        }

        return population;
    }

    public static double Ackley1Function(double[] x)
    {
        return -20 * Math.Exp(-0.02 * Math.Sqrt((1 / (x.Length * 1.0)) * x.Sum(x_i => x_i * x_i))) - Math.Exp((1 / (x.Length * 1.0)) * x.Sum(x_i => Math.Cos(2 * Math.PI * x_i))) + 20 + Math.E;
    }

    public static double Schwefel1_2Function(double[] x)
    {
        double fitness = 0;

        for(int i = 0; i < x.Length; i++)
        {
            double innerSum = 0;

            for(int j = 0; j <= i; j++)
            {
                innerSum += x[j];
            }

            fitness += Math.Pow(innerSum, 2);
        }

        return fitness;
    }

    public static double[] coinFlipCrossover(double[] parent1, double[] parent2, Func<double[], double> fitnessFunc)
    {
        Random random = new Random();

        double[] child = new double[parent1.Length];

        for (int i = 0; i < parent1.Length; i++)
        {
            double rnd = random.Next(0, 2);

            if(rnd == 0) child[i] = parent1[i];
            else child[i] = parent2[i];
        }

        return child;
    }

    // Computational Intelligence page 155
    public static double[] uniformMutation(double[] offspring, double[] lowerBound, double[] upperBound, Func<double[], double> fitnessFunc, bool minimisation = true)
    {
        Random random = new Random();
        //Console.WriteLine(random.NextDouble());

        int dimensions = offspring.Length;

        double[] mutated = new double[dimensions];

        for(int i = 0; i < dimensions; i++)
        {
            int rnd = random.Next(0, 2);

            if(rnd == 0)
            {
                mutated[i] = offspring[i] + random.NextDouble() * (upperBound[i] - offspring[i]);
            }
            else
            {
                mutated[i] = offspring[i] + random.NextDouble() * (offspring[i] - lowerBound[i]);
            }

            mutated[i] = Math.Max(offspring[i], lowerBound[i]);
            mutated[i] = Math.Min(offspring[i], upperBound[i]);
        }

        return mutated;
    }

    public static double[] simpleMutation(double[] individual, double[] lowerBound, double[] upperBound, Func<double[], double> fitnessFunc, bool minimisation = true)
    {
        Random random = new Random();

        double mutationRate = 0.15;

        for (int i = 0; i < individual.Length; i++)
        {
            // Apply mutation with probability mutationRate
            if (random.NextDouble() < mutationRate)
            {
                // Generate a random perturbation within the specified range
                double mutation = lowerBound[i] + random.NextDouble() * (upperBound[i] - lowerBound[i]) * 1;

                // Apply the mutation to the gene
                individual[i] += mutation;

                // Apply boundary constraints
                individual[i] = Math.Max(individual[i], lowerBound[i]);
                individual[i] = Math.Min(individual[i], upperBound[i]);
            }
        }

        return individual;
    }

    // Gradient Descent
    public static double[] GradientDescentMutation(double[] offspring, double[] lowerBound, double[] upperBound, Func<double[], double> fitnessFunc, bool minimisation = true)
    {
        int dimensions = offspring.Length;

        double[] mutated = (double[])offspring.Clone();

        double[] gradient = ApproximateGradient(fitnessFunc, offspring);

        for(int i = 0; i < dimensions; i++)
        {
            mutated[i] -= 0.01 * gradient[i];

            // Apply boundary constraints
            mutated[i] = Math.Min(Math.Max(mutated[i], lowerBound[i]), upperBound[i]);
        }

        return mutated;
    }

    // AdaGrad with 5 solutions choose best
    public static double[] AdaGradMutation(double[] offspring, double[] lowerBound, double[] upperBound, Func<double[], double> fitnessFunc, bool minimisation = true)
    {
        double[] mutated = AdaGradOptimiser(fitnessFunc, offspring, learningRate, lowerBound, upperBound, 3, minimisation);

        return mutated;
    }

    // AdaGrad optimiser with boundary constraints
    public static double[] AdaGradOptimiser(Func<double[], double> fitnessFunc, double[] offspring, double learningRate, double[] lowerBound, double[] upperBound, int maxIterations, bool minimisation, double epsilon = 1e-8)
    {
        double[] currentSolution = (double[])offspring.Clone();
        double[] squaredGradientSum = new double[offspring.Length];
        Array.Fill(squaredGradientSum, 0);

        for(int i = 0; i < maxIterations; i++)
        {
            double[] gradient = ApproximateGradient(fitnessFunc, currentSolution);

            for (int j = 0; j < currentSolution.Length; j++)
            {
                squaredGradientSum[j] += gradient[j] * gradient[j];

                // Compute the adaptive learning rate
                double adaptiveLearningRate = learningRate / Math.Sqrt(squaredGradientSum[j] + epsilon);

                // Update the parameter
                currentSolution[j] -= adaptiveLearningRate * gradient[j];

                // Apply boundary constraints
                currentSolution[j] = Math.Min(Math.Max(currentSolution[j], lowerBound[j]), upperBound[j]);
            }
        }
        
        return currentSolution;
    }

    public static double norm(double[] x, double[] y)
    {
        double norm = 0;
        for (int i = 0; i < x.Length; i++)
        {
            norm += Math.Pow(x[i] - y[i], 2);
        }
        norm = Math.Sqrt(norm);

        return norm;
    }

    public static double[] AdamMutation(double[] offspring, double[] lowerBound, double[] upperBound, Func<double[], double> fitnessFunc, bool minimisation = true)
    {
        int dimensions = offspring.Length;

        double[] mutated = new double[dimensions];

        mutated = AdamOptimiser(fitnessFunc, offspring, learningRate, 1, lowerBound, upperBound);

        return mutated;
    }

    // Adam optimiser for adaptive learning rate
    public static double[] AdamOptimiser(Func<double[], double> fitnessFunc, double[] offspring, double learningRate, int maxIterations, double[] lowerBound, double[] upperBound, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
    {
        int dimensions = offspring.Length;

        double[] currentSolution = (double[])offspring.Clone();
        // First moment estimate:
        double[] m = new double[dimensions];
        // Second moment estimate:
        double[] v = new double[dimensions]; 

        for (int i = 0; i < maxIterations; i++)
        {
            double[] gradient = ApproximateGradient(fitnessFunc, currentSolution);

            for (int j = 0; j < dimensions; j++)
            {
                // Update biased first moment estimate
                m[j] = beta1 * m[j] + (1 - beta1) * gradient[j];
                // Update biased second raw moment estimate
                v[j] = beta2 * v[j] + (1 - beta2) * Math.Pow(gradient[j], 2);
            
                // Bias-corrected first moment estimate
                double mHat = m[j] / (1 - Math.Pow(beta1, i + 1));
                // Bias-corrected second raw moment estimate
                double vHat = v[j] / (1 - Math.Pow(beta2, i + 1));

                // Update parameters
                currentSolution[j] -= learningRate * mHat / (Math.Sqrt(vHat) + epsilon);

                // Apply boundary constraints
                currentSolution[j] = Math.Min(Math.Max(currentSolution[j], lowerBound[j]), upperBound[j]);
            }
        }

        return currentSolution;
    }

    public static double[] AdaDeltaMutation(double[] offspring, double[] lowerBound, double[] upperBound, Func<double[], double> fitnessFunc, bool minimisation = true)
    {
        int dimensions = offspring.Length;

        double[] mutated = new double[dimensions];

        mutated = AdamOptimiser(fitnessFunc, offspring, learningRate, 1, lowerBound, upperBound);

        return mutated;
    }

    // Adam optimiser for adaptive learning rate
    public static double[] AdaDeltaOptimiser(Func<double[], double> fitnessFunc, double[] offspring, double learningRate, int maxIterations, double[] lowerBound, double[] upperBound, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
    {
        int dimensions = offspring.Length;

        double[] currentSolution = (double[])offspring.Clone();
        // First moment estimate:
        double[] m = new double[dimensions];
        // Second moment estimate:
        double[] v = new double[dimensions];

        for (int i = 0; i < maxIterations; i++)
        {
            double[] gradient = ApproximateGradient(fitnessFunc, currentSolution);

            for (int j = 0; j < dimensions; j++)
            {
                // Update biased first moment estimate
                m[j] = beta1 * m[j] + (1 - beta1) * gradient[j];
                // Update biased second raw moment estimate
                v[j] = beta2 * v[j] + (1 - beta2) * Math.Pow(gradient[j], 2);

                // Bias-corrected first moment estimate
                double mHat = m[j] / (1 - Math.Pow(beta1, i + 1));
                // Bias-corrected second raw moment estimate
                double vHat = v[j] / (1 - Math.Pow(beta2, i + 1));

                // Update parameters
                currentSolution[j] -= learningRate * mHat / (Math.Sqrt(vHat) + epsilon);

                // Apply boundary constraints
                currentSolution[j] = Math.Min(Math.Max(currentSolution[j], lowerBound[j]), upperBound[j]);
            }
        }

        return currentSolution;
    }

    public static double[] RMSPropMutation(double[] offspring, double[] lowerBound, double[] upperBound, Func<double[], double> fitnessFunc, bool minimisation = true)
    {
        int dimensions = offspring.Length;

        double[] mutated = new double[dimensions];

        Random random = new Random();

        mutated = RMSPropOptimiser(fitnessFunc, offspring, learningRate, 1, lowerBound, upperBound);

        return mutated;
    }

    // Adam optimiser for adaptive learning rate
    public static double[] RMSPropOptimiser(Func<double[], double> fitnessFunc, double[] offspring, double learningRate, int maxIterations, double[] lowerBound, double[] upperBound, double beta = 0.9, double epsilon = 1e-8)
    {
        int dimensions = offspring.Length;

        double[] currentSolution = (double[])offspring.Clone();

        double[] sumSquaredGradients = new double[dimensions];

        for (int i = 0; i < maxIterations; i++)
        {
            double[] gradient = ApproximateGradient(fitnessFunc, currentSolution);

            for (int j = 0; j < dimensions; j++)
            {
                sumSquaredGradients[j] *= beta + (1 - beta) * Math.Pow(gradient[j], 2);

                // Update parameters
                currentSolution[j] -= (learningRate / (Math.Sqrt(sumSquaredGradients[j])) * gradient[j]);

                // Apply boundary constraints
                currentSolution[j] = Math.Min(Math.Max(currentSolution[j], lowerBound[j]), upperBound[j]);
            }
        }

        return currentSolution;
    }

    // Approximate gradient using finite differences
    public static double[] ApproximateGradient(Func<double[], double> fitnessFunc, double[] x, double epsilon = 1e-6)
    {
        double[] gradient = new double[x.Length];

        for (int i = 0; i < x.Length; i++)
        {
            double[] xPlusEps = (double[])x.Clone();
            double[] xMinusEps = (double[])x.Clone();

            xPlusEps[i] += epsilon;
            xMinusEps[i] -= epsilon;

            gradient[i] = (fitnessFunc(xPlusEps) - fitnessFunc(xMinusEps)) / (2 * epsilon);
        }

        return gradient;
    }

    public static double[] tournamentSelection(double[][] population, int populationSize, Func<double[], double> fitnessFunc, bool minimisation = true)
    {
        int tournamentSize = (int) Math.Ceiling(populationSize * 0.8);

        double[][] tournament = new double[tournamentSize][];

        Random random = new Random();

        int rnd = random.Next(0, populationSize);

        for (int i = 0; i < tournamentSize; i++)
        {
            tournament[i] = population[rnd];
        }

        Array.Sort(tournament, (x, y) =>
        {
            if (minimisation)
                return fitnessFunc(x).CompareTo(fitnessFunc(y));
            else
                return fitnessFunc(y).CompareTo(fitnessFunc(x));
        });

        return tournament[0];
    }

    public static double[][] select2(double[][] population, int populationSize, Func<double[], double> fitnessFunc, bool minimisation = true)
    { 
        return [tournamentSelection(population, populationSize, fitnessFunc, minimisation), tournamentSelection(population, populationSize, fitnessFunc, minimisation)];
    }

    public static double[] GA(double[] ga, double[][] population, Func<double[], double> fitnessFunc, double[] lowerBound, double[] upperBound, Func<double[], double[], double[], Func<double[], double>, bool, double[]> mutationFunc, bool minimisation = true, double error = 0.01, int maxIterations = int.MaxValue)
    {
        //double[][] population = initialisePopulation(lowerBound, upperBound, populationSize);
        var stopwatch = new Stopwatch();
        stopwatch.Start();

        Array.Sort(population, (x, y) =>
        {
            if(minimisation)
                return fitnessFunc(x).CompareTo(fitnessFunc(y));
            else
                return fitnessFunc(y).CompareTo(fitnessFunc(x));
        });

        int iterations = 0;

        while (iterations < 100)
        {
            ga[iterations] += fitnessFunc(population[0]);

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
                if (minimisation)
                    return fitnessFunc(x).CompareTo(fitnessFunc(y));
                else
                    return fitnessFunc(y).CompareTo(fitnessFunc(x));
            });

        }

        stopwatch.Stop();

        Console.WriteLine("Iterations: " + iterations);
        Console.WriteLine("Fitness of fittest individual: " + fitnessFunc(population[0]));
        Console.WriteLine("Fittest individual: [" + string.Join(", ", population[0]) + "]");

        return [iterations, fitnessFunc(population[0]), stopwatch.Elapsed.TotalMilliseconds];
    }

    /*static public void Main(String[] args)
    {
        int n = 100;
        int iterations = 300;
        double[] lowerBound = [-35, -35, -35, -35, -35];
        double[] upperBound = [35, 35, 35, 35, 35];
        //double[] lowerBound = [-100, -100, -100, -100, -100];
        //double[] upperBound = [100, 100, 100, 100, 100];
        Func<double[], double> fitnessFunc = Ackley1Function;
        Func<double[], double[], double[], Func<double[], double>, bool, double[]> mutationFunc = GradientDescentMutation;

        for (int i = 0; i < 10; i++)
        {

            KeyValuePair<double[], double[][]>[] population = Adam.initialisePopulation(lowerBound, upperBound, n);

           
            double[][] normalPop = new double[population.Length][];
            KeyValuePair < double[], double[]>[] pairedPop = new KeyValuePair<double[], double[]>[population.Length];

            for (int j = 0; j < population.Length; j++)
            {
                normalPop[j] = (double[])population[j].Key.Clone();
                pairedPop[j] = new KeyValuePair<double[], double[]>((double[])normalPop[j].Clone(), (double[])population[j].Value[0].Clone());
            }

            learningRate = 0.6;

            double[] resultSet = GAwithPairing.GA(ga1, (KeyValuePair<double[], double[]>[])pairedPop.Clone(), fitnessFunc, lowerBound, upperBound, GAwithPairing.RMSPropMutation);

            resultSet = GA(ga2, (double[][])normalPop.Clone(), fitnessFunc, lowerBound, upperBound, simpleMutation);

            resultSet = GA(ga3, normalPop, fitnessFunc, lowerBound, upperBound, GradientDescentMutation);

            resultSet = Adam.GA(ga4, population, fitnessFunc, lowerBound, upperBound, Adam.AdamMutation);
        }
        string path = @"Results.txt";
        using (StreamWriter sw = File.CreateText(path))
        {
            sw.WriteLine("lr1 = -1*[" + String.Join(" ", ga3) + "]./10;");
            sw.WriteLine("lr2 = -1*[" + String.Join(" ", ga2) + "]./10;");
            sw.WriteLine("lr3 = -1*[" + String.Join(" ", ga1) + "]./10;");
            sw.WriteLine("lr4 = -1*[" + String.Join(" ", ga4) + "]./10;");
        }

            /*
            for (int j = 0; j < 10; j++)
            {
                KeyValuePair<double[], double[][]>[] populationWithLearningRates = GAwithPairing.initialisePopulation(lowerBound, upperBound, n, 35);

                double[][] population = new double[n][];

                for (int i = 0; i < n; i++)
                {
                    population[i] = (double[])populationWithLearningRates[i].Key.Clone();
                }



                var stopwatch = new Stopwatch();
                stopwatch.Start();

                double[] one = GA(ga1, (double[][])population.Clone(), Ackley1Function, lowerBound, upperBound, simpleMutation, true, 0.01, iterations);

                stopwatch.Stop();

                Console.WriteLine("Time elapsed: " + stopwatch.Elapsed.TotalMilliseconds);
                Console.WriteLine();

                stopwatch.Reset();

                stopwatch.Start();

                double[] two = GA(ga2, (double[][])population.Clone(), Ackley1Function, lowerBound, upperBound, AdaGradMutation1, true, 0.01, iterations);

                stopwatch.Stop();

                Console.WriteLine("Time elapsed: " + stopwatch.Elapsed.TotalMilliseconds);
                Console.WriteLine();

                stopwatch.Reset();

                stopwatch.Start();

                double[] three = GAwithPairing.GA(ga3, (KeyValuePair<double[], double[][]>[])populationWithLearningRates.Clone(), Ackley1Function, lowerBound, upperBound, GAwithPairing.RMSPropMutation, true, 0.01, iterations);

                stopwatch.Stop();

                Console.WriteLine("Time elapsed: " + stopwatch.Elapsed.TotalMilliseconds);
                Console.WriteLine();

                stopwatch.Reset();

                stopwatch.Start();

                double[] four = GAwithPairing.GA(ga4, populationWithLearningRates, Ackley1Function, lowerBound, upperBound, GAwithPairing.AdaGradMutation, true, 0.01, iterations);



                stopwatch.Stop();

                Console.WriteLine("Time elapsed: " + stopwatch.Elapsed.TotalMilliseconds);
                Console.WriteLine();
            }


            string path = @"Results.txt";
                // Create a file to write to.
                using (StreamWriter sw = File.CreateText(path))
                {
                    sw.WriteLine("simple = -1*[" + String.Join(" ", ga1) + "]./10;");
                    sw.WriteLine("adagrad3 = -1*[" + String.Join(" ", ga2) + "]./10;");
                    sw.WriteLine("rmsproppair = -1*[" + String.Join(" ", ga3) + "]./10;");
                    sw.WriteLine("adagradpair = -1*[" + String.Join(" ", ga4) + "]./10;");
                }
            */
            /*
             for (int i = 0; i < 10; i++)
            {
                double[][] population = initialisePopulation(lowerBound, upperBound, n);

                var stopwatch = new Stopwatch();

                mutationFunc = uniformMutation;

                double[] learningRates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0];

                stopwatch.Start();
                learningRate = learningRates[0];
                double[] one = GA(ga1, (double[][])population.Clone(), fitnessFunc, lowerBound, upperBound, mutationFunc, true, 0.01, iterations);

                stopwatch.Stop();
                stopwatch.Reset();

                mutationFunc = GradientDescentMutation;

                stopwatch.Start();
                learningRate = learningRates[2];
                double[] two = GA(ga2, (double[][])population.Clone(), fitnessFunc, lowerBound, upperBound, mutationFunc, true, 0.01, iterations);

                stopwatch.Stop();
                stopwatch.Reset();

                /*stopwatch.Start();
                learningRate = learningRates[2];
                double[] three = GA(ga3, (double[][])population.Clone(), fitnessFunc, lowerBound, upperBound, mutationFunc, true, 0.01, iterations);

                stopwatch.Stop();
                stopwatch.Reset();

                mutationFunc = uniformMutation;

                stopwatch.Start();
                learningRate = learningRates[3];
                double[] four = GA(ga4, (double[][])population.Clone(), fitnessFunc, lowerBound, upperBound, mutationFunc, true, 0.01, iterations);

                stopwatch.Stop();
                stopwatch.Reset();

                stopwatch.Start();
                learningRate = learningRates[4];
                double[] five = GA(ga5, (double[][])population.Clone(), fitnessFunc, lowerBound, upperBound, mutationFunc, true, 0.01, iterations);

                stopwatch.Stop();
                stopwatch.Reset();

                stopwatch.Start();
                learningRate = learningRates[5];
                double[] six = GA(ga6, (double[][])population.Clone(), fitnessFunc, lowerBound, upperBound, mutationFunc, true, 0.01, iterations);

                stopwatch.Stop();
                stopwatch.Reset();
            }

            string path = @"Results.txt";
            // Create a file to write to.
            using (StreamWriter sw = File.CreateText(path))
            {
                sw.WriteLine("lr1 = -1*[" + String.Join(" ", ga1) + "]./10;");
                sw.WriteLine("lr2 = -1*[" + String.Join(" ", ga2) + "]./10;");
               /* sw.WriteLine("lr3 = -1*[" + String.Join(" ", ga3) + "]./10;");
                sw.WriteLine("lr4 = -1*[" + String.Join(" ", ga4) + "]./10;");
                sw.WriteLine("lr5 = -1*[" + String.Join(" ", ga5) + "]./10;");
                sw.WriteLine("lr6 = -1*[" + String.Join(" ", ga6) + "]./10;");
            }*/

        
}
