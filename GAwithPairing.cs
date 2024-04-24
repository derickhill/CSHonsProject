using MathNet.Numerics;
using MathNet.Numerics.Distributions;
using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.ExceptionServices;

namespace CSProject
{
    public class GAwithPairing
    {
        public static KeyValuePair<double[], double[]>[] initialisePopulation(double[] lowerBound, double[] upperBound, int n)
        {
            int dimensions = lowerBound.Length;

            KeyValuePair <double[], double[]>[] populationWithLearningRates = new KeyValuePair<double[], double[]>[n];

            Random random = new Random();

            double[] squaredGradientSums = new double[dimensions];
            Array.Fill(squaredGradientSums, 0);

            for (int i = 0; i < n; i++)
            {

                double[] curIndividual = new double[dimensions];

                for (int j = 0; j < dimensions; j++)
                {
                    curIndividual[j] = random.NextDouble() * (upperBound[j] - lowerBound[j]) + lowerBound[j];
                }

                populationWithLearningRates[i] = new KeyValuePair<double[], double[]>(curIndividual, (double[])squaredGradientSums.Clone());
            }

            return populationWithLearningRates;
        }

        public static double Ackley1Function(double[] x)
        {
            return -20 * Math.Exp(-0.02 * Math.Sqrt((1 / (x.Length * 1.0)) * x.Sum(x_i => x_i * x_i))) - Math.Exp((1 / (x.Length * 1.0)) * x.Sum(x_i => Math.Cos(2 * Math.PI * x_i))) + 20 + Math.E;
        }
        
        public static KeyValuePair<double[], double[]> coinFlipCrossover(KeyValuePair<double[], double[]> parent1, KeyValuePair<double[], double[]> parent2)
        {
            Random random = new Random();

            int dimensions = parent1.Key.Length;

            double[] child = new double[dimensions];
            double[] childSquaredGradientSums = new double[dimensions];

            for (int i = 0; i < dimensions; i++)
            {
                double rnd = random.Next(0, 2);

                if (rnd == 0)
                {
                    child[i] = parent1.Key[i];
                    childSquaredGradientSums[i] = parent1.Value[i];
                }
                else
                {
                    child[i] = parent2.Key[i];
                    childSquaredGradientSums[i] = parent2.Value[i];
                }
            }

            return new KeyValuePair<double[], double[]>(child, childSquaredGradientSums);
        }

        //RMSProp
        public static KeyValuePair<double[], double[]> RMSPropMutation(KeyValuePair<double[], double[]> offspring, double[] lowerBound, double[] upperBound, Func<double[], double> fitnessFunc, bool minimisation = true)
        {
            KeyValuePair<double[], double[]> mutated = RMSPropOptimiser(fitnessFunc, offspring, lowerBound, upperBound);

            return mutated;
        }

        public static KeyValuePair<double[], double[]> RMSPropOptimiser(Func<double[], double> fitnessFunc, KeyValuePair<double[], double[]> offspring, double[] lowerBound, double[] upperBound, double epsilon = 1e-8)
        {
            double[] individual = (double[])offspring.Key.Clone();
            double[] squaredGradientSums = (double[])offspring.Value.Clone();
            double[] gradient = Project.ApproximateGradient(fitnessFunc, individual);

            double beta = 0.9;

            for(int i = 0; i < squaredGradientSums.Length; i++)
            {
                squaredGradientSums[i] = beta * squaredGradientSums[i] + (1 - beta) * Math.Pow(gradient[i], 2);

                individual[i] -= (1.2/ Math.Sqrt(squaredGradientSums[i]) + epsilon) * gradient[i];


                individual[i] = Math.Min(Math.Max(individual[i], lowerBound[i]), upperBound[i]);
            }

            return new KeyValuePair<double[], double[]>(individual, squaredGradientSums);
        }

        // AdaGrad
        public static KeyValuePair<double[], double[]> AdaGradMutation(KeyValuePair<double[], double[]> offspring, double[] lowerBound, double[] upperBound, Func<double[], double> fitnessFunc, bool minimisation = true)
        {
            KeyValuePair<double[], double[]> mutated = AdaGradOptimiser(fitnessFunc, offspring, lowerBound, upperBound);

            return mutated;
        }

        // AdaGrad optimiser with boundary constraints
        public static KeyValuePair<double[], double[]> AdaGradOptimiser(Func<double[], double> fitnessFunc, KeyValuePair<double[], double[]> offspring, double[] lowerBound, double[] upperBound, double epsilon = 1e-8)
        {
            double[] individual = (double[])offspring.Key.Clone();
            double[] squaredGradientSums = (double[])offspring.Value.Clone();
            double[] gradient = Project.ApproximateGradient(fitnessFunc, individual);

            for (int i = 0; i < individual.Length; i++)
            {
                squaredGradientSums[i] += gradient[i] * gradient[i];

                // Compute the adaptive learning rate
                double adaptiveLearningRate = 5 / Math.Sqrt(squaredGradientSums[i] + epsilon);

                /*if(adaptiveLearningRate < epsilon)
                {
                    squaredGradientSums[i] = 0;
                }*/

                //Console.WriteLine(gradient[i]);

                // Update the parameter
                individual[i] -= adaptiveLearningRate * gradient[i];

                // Apply boundary constraints
                individual[i] = Math.Min(Math.Max(individual[i], lowerBound[i]), upperBound[i]);
            }


            return new KeyValuePair<double[], double[]>(individual, squaredGradientSums);
        }

        // Approximate gradient using finite differences
        public static double[] ApproximateGradient(Func<double[], double> fitnessFunc, double[] x, double epsilon = 1)
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

        public static KeyValuePair<double[], double[]> tournamentSelection(KeyValuePair<double[], double[]>[] population, int populationSize, Func<double[], double> fitnessFunc, bool minimisation = true)
        {
            int tournamentSize = (int)Math.Ceiling(populationSize * 0.8);

            KeyValuePair<double[], double[]>[] tournament = new KeyValuePair<double[], double[]>[tournamentSize];

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

        public static KeyValuePair<double[], double[]>[] select2(KeyValuePair<double[], double[]>[] population, int populationSize, Func<double[], double> fitnessFunc, bool minimisation = true)
        {
            return [tournamentSelection(population,populationSize, fitnessFunc, minimisation), tournamentSelection(population, populationSize, fitnessFunc, minimisation)];
        }

        public static double[] GA(double[] ga, KeyValuePair<double[], double[]>[] population, Func<double[], double> fitnessFunc, double[] lowerBound, double[] upperBound, Func<KeyValuePair<double[], double[]>, double[], double[], Func<double[], double>, bool, KeyValuePair<double[], double[]>> mutationFunc, bool minimisation = true, double error = 0.01, int maxIterations = int.MaxValue)
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
                ga[iterations] += fitnessFunc(population[0].Value);

                iterations++;

                KeyValuePair<double[], double[]>[] tempPopulation = new KeyValuePair<double[], double[]>[population.Length];
                int tempPopulationSize = 0;

                tempPopulation[0] = population[0];
                tempPopulationSize++;

                while (tempPopulationSize < population.Length)
                {
                    KeyValuePair<double[], double[]>[] parents = select2(tempPopulation, tempPopulationSize, fitnessFunc, minimisation);

                    KeyValuePair<double[], double[]> crossedOver = coinFlipCrossover(parents[0], parents[1]);

                    KeyValuePair<double[], double[]> mutated = mutationFunc(crossedOver, lowerBound, upperBound, fitnessFunc, minimisation);

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
}