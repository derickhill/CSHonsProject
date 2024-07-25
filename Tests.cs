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
            int product = 5 * 10 * 5;
            int counter = 0;
            double percentage = 0;

            int maxFunctionCalls = 100000;
            int n = 100;

            double[] functionCalls;

            string path;
            string folder = "Results/";

            for (int i = 0; i < 5; i++)
            {
                Console.WriteLine("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^");
                Console.WriteLine("Iterations:" + i);
                Console.WriteLine("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^");

                // Fitness functions
                for (int j = 0; j < 10; j++)
                {
                    Console.WriteLine("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_");
                    Console.WriteLine("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_");
                    Console.WriteLine("Fitness function " + j);
                    Console.WriteLine("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_");
                    Console.WriteLine("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_");

                    FitnessFunc fitnessFunc = new(j, maxFunctionCalls);

                    int[] dimensions = [5, 10, 20, 40, 80];

                    foreach (int dim in dimensions)
                    {
                        Console.WriteLine("--------------------------------------------------------------------------------------");
                        Console.WriteLine("Dimensions: " + dim);
                        Console.WriteLine("--------------------------------------------------------------------------------------");

                        double[] learningRates = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1.6, 2];
                        double[] epsilons = [1e-6, 0.0001, 0.001, 0.01, 0.1, 1, 2];
                        double[] betas = [0.6, 0.8, 0.9, 0.95, 0.999];
                        int p = 0;

                        double[] oneIter = [0];
                        Algorithms.setEps(0.1);

                        double[][] population = Algorithms.initialisePopulation(fitnessFunc.getLowerBound(dim), fitnessFunc.getUpperBound(dim), n);

                        double initialLearningRate = 0.4;

                        /*foreach(double rho in betas)
                        {
                            Algorithms.setEps(0.1);
                            //Algorithms.setMethod(v);

                            Console.WriteLine("------------------------------------------");
                            Console.WriteLine("Rho:" + rho);
                            Console.WriteLine("------------------------------------------");

                            functionCalls = Algorithms.AdaDeltaGA(population, fitnessFunc, rho);
                            path = folder + "adadelta_dim" + dim + "_r" + p + "_ff" + j + "_iter" + i;
                            writeToTextFile(path, functionCalls);
                            fitnessFunc.reset();

                            Console.WriteLine("AdaDelta done");

                            /*functionCalls = Algorithms.RandomGA(population, fitnessFunc, initialLearningRate * 10);
                            path = folder + "random_dim" + dim + "_m" + v + "_ff" + j + "_iter" + i;
                            writeToTextFile(path, functionCalls);
                            fitnessFunc.reset();

                            Console.WriteLine("Random done");

                            functionCalls = Algorithms.GradientDescentGA(population, fitnessFunc, initialLearningRate);
                            path = folder + "gd_dim" + dim + "_m" + v + "_ff" + j + "_iter" + i;
                            writeToTextFile(path, functionCalls);
                            fitnessFunc.reset();

                            Console.WriteLine("GD done");

                            functionCalls = Algorithms.AdaGradGA(population, fitnessFunc, initialLearningRate);
                            path = folder + "adagrad_dim" + dim + "_m" + v + "_ff" + j + "_iter" + i;
                            writeToTextFile(path, functionCalls);
                            fitnessFunc.reset();

                            Console.WriteLine("AdaGrad done");

                            functionCalls = Algorithms.RMSPropGA(population, fitnessFunc, initialLearningRate);
                            path = folder + "rmsprop_dim" + dim + "_m" + v + "_ff" + j + "_iter" + i;
                            writeToTextFile(path, functionCalls);
                            fitnessFunc.reset();

                            Console.WriteLine("RMSProp done");

                            functionCalls = Algorithms.AdaDeltaGA(population, fitnessFunc);
                            path = folder + "adadelta_dim" + dim + "_m" + v + "_ff" + j + "_iter" + i;
                            writeToTextFile(path, functionCalls);
                            fitnessFunc.reset();

                            Console.WriteLine("AdaDelta done");

                            functionCalls = Algorithms.AdamGA(population, fitnessFunc, initialLearningRate);
                            path = folder + "adam_dim" + dim + "_m" + v + "_ff" + j + "_iter" + i;
                            writeToTextFile(path, functionCalls);
                            fitnessFunc.reset();

                            Console.WriteLine("Adam done");*/

                        /*Console.WriteLine("------------------------------------------");
                        Console.WriteLine("Initial learning rate:" + initialLearningRate);
                        Console.WriteLine("------------------------------------------");


                        functionCalls = Algorithms.RandomGA(population, fitnessFunc, initialLearningRate * 10);
                        path = folder + "random_dim" + dim + "_lr" + p + "_ff" + j + "_iter" + i;
                        writeToTextFile(path, functionCalls);
                        fitnessFunc.reset();

                        Console.WriteLine("Random done");

                        functionCalls = Algorithms.GradientDescentGA(population, fitnessFunc, initialLearningRate);
                        path = folder + "gd_dim" + dim + "_lr" + p + "_ff" + j + "_iter" + i;
                        writeToTextFile(path, functionCalls);
                        fitnessFunc.reset();

                        Console.WriteLine("GD done");

                        functionCalls = Algorithms.AdaGradGA(population, fitnessFunc, initialLearningRate);
                        path = folder + "adagrad_dim" + dim + "_lr" + p + "_ff" + j + "_iter" + i;
                        writeToTextFile(path, functionCalls);
                        fitnessFunc.reset();

                        Console.WriteLine("AdaGrad done");

                        functionCalls = Algorithms.RMSPropGA(population, fitnessFunc, initialLearningRate);
                        path = folder + "rmsprop_dim" + dim + "_lr" + p  + "_ff" + j + "_iter" + i;
                        writeToTextFile(path, functionCalls);
                        fitnessFunc.reset();

                        Console.WriteLine("RMSProp done");

                        functionCalls = Algorithms.AdaDeltaGA(population, fitnessFunc);
                        path = folder + "adadelta_dim" + dim + "_lr" + p + "_ff" + j + "_iter" + i;
                        writeToTextFile(path, functionCalls);
                        fitnessFunc.reset();

                        Console.WriteLine("AdaDelta done");

                        functionCalls = Algorithms.AdamGA(population, fitnessFunc, initialLearningRate);
                        path = folder + "adam_dim" + dim + "_lr" + p + "_ff" + j + "_iter" + i;
                        writeToTextFile(path, functionCalls);
                        fitnessFunc.reset();

                        Console.WriteLine("Adam done");


                        percentage++;
                        double percent = Math.Floor(percentage / product * 100);

                        string loadingBar = "|";

                        for (int f = 0; f < percent; f++)
                        {
                            loadingBar += "=";
                        }

                        for (int g = 0; g < 100 - (int)percent; g++)
                        {
                            loadingBar += " ";
                        }

                        loadingBar += "|";

                            Console.WriteLine((loadingBar) + percent + "%");


                        p++;
                    }*/

                        functionCalls = Algorithms.AdaDeltaGA(population, fitnessFunc);
                        path = folder + "adadelta_dim" + dim + "_ff" + j + "_iter" + i;
                        writeToTextFile(path, functionCalls);
                        fitnessFunc.reset();

                        Console.WriteLine("AdaDeltaGA done");

                        functionCalls = Algorithms.AdaDelta(population, fitnessFunc);
                        path = folder + "sadadelta_dim" + dim + "_ff" + j + "_iter" + i;
                        writeToTextFile(path, functionCalls);
                        fitnessFunc.reset();

                        Console.WriteLine("AdaDelta done");

                        functionCalls = Algorithms.AdaGradGA(population, fitnessFunc, initialLearningRate);
                        path = folder + "adagrad_dim" + dim + "_ff" + j + "_iter" + i;
                        writeToTextFile(path, functionCalls);
                        fitnessFunc.reset();

                        Console.WriteLine("AdaGradGA done");

                        functionCalls = Algorithms.AdaGrad(population, fitnessFunc, initialLearningRate);
                        path = folder + "sadagrad_dim" + dim + "_ff" + j + "_iter" + i;
                        writeToTextFile(path, functionCalls);
                        fitnessFunc.reset();

                        Console.WriteLine("AdaGrad done");

                        functionCalls = Algorithms.AdamGA(population, fitnessFunc, initialLearningRate);
                        path = folder + "adam_dim" + dim + "_ff" + j + "_iter" + i;
                        writeToTextFile(path, functionCalls);
                        fitnessFunc.reset();

                        Console.WriteLine("AdamGA done");

                        functionCalls = Algorithms.Adam(population, fitnessFunc, initialLearningRate);
                        path = folder + "sadam_dim" + dim + "_ff" + j + "_iter" + i;
                        writeToTextFile(path, functionCalls);
                        fitnessFunc.reset();

                        Console.WriteLine("Adam done");

                        functionCalls = Algorithms.RMSPropGA(population, fitnessFunc, initialLearningRate);
                        path = folder + "rmsprop_dim" + dim + "_ff" + j + "_iter" + i;
                        writeToTextFile(path, functionCalls);
                        fitnessFunc.reset();

                        Console.WriteLine("RMSPropGA done");

                        functionCalls = Algorithms.RMSProp(population, fitnessFunc, initialLearningRate);
                        path = folder + "srmsprop_dim" + dim + "_ff" + j + "_iter" + i;
                        writeToTextFile(path, functionCalls);
                        fitnessFunc.reset();

                        Console.WriteLine("RMSProp done");

                        Console.WriteLine("*****************************************************");
                        Console.WriteLine((++counter) + " / " + product);
                        Console.WriteLine("*****************************************************");
                    }

                }


            }

            //MLApp.MLApp matlab = new MLApp.MLApp();

            //matlab.Execute(folder + "projectplots1.m");
        }
    }
}
