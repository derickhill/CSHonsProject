using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using MathWorks.MATLAB.Engine;
using MathWorks.MATLAB.Exceptions;
using MathWorks.MATLAB.Types;


namespace CSProject
{
    class Tests
    {
        private static void writeToTextFile(string path, double[] functionCalls)
        {
            string jsonString = JsonSerializer.Serialize(functionCalls);
            File.WriteAllText(path + ".json", jsonString);
        }

        public static void Main(string[] args)
        {
            int product = 15 * 10 * 6 * 20;
            double percentage = 0;

            int maxFunctionCalls = 50000;
            int n = 100;

            double[] functionCalls;

            string path;
            string folder = "Results/";

            for (int i = 0; i < 15; i++)
            {
                // Fitness functions
                for (int j = 0; j < 10; j++)
                {
                    Console.WriteLine("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_");
                    Console.WriteLine("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_");
                    Console.WriteLine("Fitness function " + j);
                    Console.WriteLine("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_");
                    Console.WriteLine("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_");

                    FitnessFunc fitnessFunc = new(j, maxFunctionCalls);

                    int[] dimensions = [5, 10, 20, 40, 50, 80];

                    foreach (int dim in dimensions)
                    {
                        Console.WriteLine("--------------------------------------------------------------------------------------");
                        Console.WriteLine("Dimensions: " + dim);
                        Console.WriteLine("--------------------------------------------------------------------------------------");

                        double[] learningRates = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.3, 1.6, 1.7, 1.8, 1.9, 2];

                        int p = 0;

                        double[][] population = Algorithms.initialisePopulation(fitnessFunc.getLowerBound(dim), fitnessFunc.getUpperBound(dim), n);

                        foreach (double learningRate in learningRates)
                        {
                            Console.WriteLine("------------------------------------------");
                            Console.WriteLine("Learning rate:" + learningRate);
                            Console.WriteLine("------------------------------------------");

                            functionCalls = Algorithms.RandomGA(population, fitnessFunc);
                            fitnessFunc.reset();
                            path = folder + "random_dim" + dim + "_lr" + p + "_ff" + j + "_iter" + i;
                            writeToTextFile(path, functionCalls);

                            Console.WriteLine("Random done");
 /*                           
                            functionCalls = Algorithms.GradientDescentGA(population, fitnessFunc, learningRate);
                            path = folder + "gd_dim" + dim + "_lr" + p + "_ff" + j + "_iter" + i;
                            writeToTextFile(path, functionCalls);
                            fitnessFunc.reset();
                            
                            Console.WriteLine("GD done");
  */                         
                            functionCalls = Algorithms.AdaGradGA(population, fitnessFunc, learningRate);
                            path = folder + "adagrad_dim" + dim + "_lr" + p + "_ff" + j + "_iter" + i;
                            writeToTextFile(path, functionCalls);
                            fitnessFunc.reset();
                            
                            Console.WriteLine("AdaGrad done");

                             functionCalls = Algorithms.RMSPropGA(population, fitnessFunc, learningRate);
                            path = folder + "rmsprop_dim" + dim + "_lr" + p + "_ff" + j + "_iter" + i;
                             writeToTextFile(path, functionCalls);
                            fitnessFunc.reset();
                            
                            Console.WriteLine("RMSProp done");

                            functionCalls = Algorithms.AdamGA(population, fitnessFunc, learningRate);
                            path = folder + "adam_dim" + dim + "_lr" + p + "_ff" + j + "_iter" + i;
                            writeToTextFile(path, functionCalls);
                            fitnessFunc.reset();
                            Console.WriteLine("Adam done");

                            p++;

                            percentage++;
                            double percent = Math.Floor(percentage / product * 100);

                            string loadingBar = "|";

                            for(int f = 0; f < percent; f++)
                            {
                                loadingBar += "=";
                            }

                            for(int g = 0; g < 100 - (int)percent; g++)
                            {
                                loadingBar += " ";
                            }

                            loadingBar += "|";

                            Console.WriteLine((loadingBar) + percent + "%");
                        }
                    }

                }


            }

            MLApp.MLApp matlab = new MLApp.MLApp();

            matlab.Execute(folder + "projectplots1.m");
        }
    }
}
