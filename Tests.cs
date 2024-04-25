using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSProject
{
    class Tests
    {
        private static void makeTextFile(string path)
        {
            // Create a file to write to.
            using (StreamWriter sw = File.CreateText(path))
            {
                sw.WriteLine("%DateTime: " + DateTime.Now.ToString());
            }
        }

        private static void writeToTextFile(string path, double[] functionCalls)
        {
            using (StreamWriter sw = new StreamWriter(path, true))
            {
                foreach(double call in functionCalls)
                {
                    sw.WriteLine(call.ToString());
                }
            }
        }

        public static void Main(string[] args)
        {
            int maxFunctionCalls = 50000;
            int n = 100;

            double[] functionCalls;

            string path;
            string folder = "Results/";

            // Fitness functions
            for(int j = 0; j < 10; j++)
            {
                Console.WriteLine("-------------------------------------------------------------");
                Console.WriteLine("Fitness function " + j);
                Console.WriteLine("-------------------------------------------------------------");

                FitnessFunc fitnessFunc = new(j, maxFunctionCalls);
                
                int[] dimensions = [5, 10, 20, 40, 50, 80];

                foreach(int dim in dimensions)
                {
                    Console.WriteLine("-------------------------------------------------------------");
                    Console.WriteLine("Dimensions: " + dim);
                    Console.WriteLine("-------------------------------------------------------------");

                    double[][][] populations = new double[15][][];
                    
                    // Generate 15 unique populations
                    for(int k = 0; k < 15; k++)
                    {
                        populations[k] = Algorithms.initialisePopulation(fitnessFunc.getLowerBound(dim), fitnessFunc.getUpperBound(dim), n);
                    }

                    double[] learningRates = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.3, 1.6, 1.7, 1.8, 1.9, 2];

                    int p = 0;

                    foreach(double learningRate in learningRates)
                    {
                        Console.WriteLine("-------------------------------------------------------------");
                        Console.WriteLine("Learning rate:" + learningRate);
                        Console.WriteLine("-------------------------------------------------------------");

     
                        for (int i = 0; i < 15; i++)
                        {
                            functionCalls = Algorithms.RandomGA(populations[i], fitnessFunc);
                            fitnessFunc.reset();
                            path = folder + "random_dim" + dim + "_lr" + p + "_ff" + j + "_iter" + i + ".txt"; ;
                            makeTextFile(path);
                            writeToTextFile(path, functionCalls);
                        }
                        Console.WriteLine("Random done");

                        for (int i = 0; i < 15; i++)
                        {
                            functionCalls = Algorithms.GradientDescentGA(populations[i], fitnessFunc, learningRate);
                            path = folder + "gd_dim" + dim + "_lr" + p + "_ff" + j + "_iter" + i + ".txt";
                            makeTextFile(path);
                            writeToTextFile(path, functionCalls);
                            fitnessFunc.reset();
                        }
                        Console.WriteLine("GD done");

                        for (int i = 0; i < 15; i++)
                        {
                            functionCalls = Algorithms.AdaGradGA(populations[i], fitnessFunc, learningRate);
                            path = folder + "adagrad_dim" + dim + "_lr" + p + "_ff" + j + "_iter" + i + ".txt"; ;
                            makeTextFile(path);
                            writeToTextFile(path, functionCalls);
                            fitnessFunc.reset();
                        }
                        Console.WriteLine("AdaGrad done");

                        for (int i = 0; i < 15; i++)
                        {
                            functionCalls = Algorithms.RMSPropGA(populations[i], fitnessFunc, learningRate);
                            path = folder + "rmsprop_dim" + dim + "_lr" + p + "_ff" + j + "_iter" + i + ".txt";
                            makeTextFile(path);
                            writeToTextFile(path, functionCalls);
                            fitnessFunc.reset();
                        }
                        Console.WriteLine("RMSProp done");

                        
                        for (int i = 0; i < 15; i++)
                        {
                            functionCalls = Algorithms.AdamGA(populations[i], fitnessFunc, learningRate);
                            path = folder + "adam_dim" + dim + "_lr" + p + "_ff" + j + "_iter" + i + ".txt";
                            makeTextFile(path);
                            writeToTextFile(path, functionCalls);
                            fitnessFunc.reset();
                        }
                        Console.WriteLine("Adam done");

                        p++;
                    }
                }
                
            }
        }
    }
}
