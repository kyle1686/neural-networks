import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Scanner;

/**
 * The A-B-C Network is a multi-layer perceptron that uses gradient descent to minimize the error across all test cases. 
 * With every iteration, it takes a 'step downhill' the error curve and optimizes its existing weights to reduce the error 
 * as much as possible for every iteration. The network is based off a feed-forward model and uses backpropgation to optimize
 * calculations.
 * 
 * @author  Kyle Li
 * @version 27 March 2024
 * Date of creation: 1 February 2024
 */
public class ABCNetwork
{
   static final int UNIT_CONVERSION_THOUSAND = 1000;
   static final int SECONDS_IN_MINUTE = 60;
   static final int MINUTES_IN_HOUR = 60;
   static final int HOURS_IN_DAY = 24;
   static final int DAYS_IN_WEEK = 7; 
   static final String DEFAULT_FILE_NAME = "./default.txt";

   int inputNodes;
   int hiddenLayerNodes;
   int outputNodes;

   double a[]; 
   double h[]; 
   double F[]; 
   
   double upperThetaJ[];
   double lowerPsi[];

   double totalError;
   double avgError; 
   double errorThreshold; 

   double weightsKJ[][];
   double weightsJI[][];

   double lambda; 

   double changeInWeightsKJ[][];
   double changeInWeightsJI[][];

   double changeInErrorKJ[][];
   double changeInErrorJI[][];

   int iterations;
   int maxIterations;

   int numberOfCases; 
   double testCases[][]; 
   double trueOutputs[][];

   double low;
   double high;

   boolean useRandomWeights;
   boolean shouldTrain;
   boolean shouldSaveWeights;

   String weightsFilePath;

   Scanner in;
   PrintWriter out;
   NNParser p; 

/**
 * Parses the given configuration file and gets the values for all the variables in it
 * 
 * @param configFilePath   the filepath to the configuration file
 * @throws FileNotFoundException may be thrown if the given file does not exist
 */
   public void parseConfigParams(String configFilePath) throws FileNotFoundException
   {
      p = new NNParser(new Scanner(new File(configFilePath)));
      p.parseConfigFile();
   }

/**
 * Sets configuration parameters, initializing variables to their respective constant values.
 */
   public void setConfigParams()
   {
      inputNodes = p.inputNodes;
      hiddenLayerNodes = p.hiddenLayerNodes;
      outputNodes = p.outputNodes;

      errorThreshold = p.errorThreshold;

      maxIterations = p.maxIterations;

      numberOfCases = p.numberOfCases;

      lambda = p.lambda;

      low = p.low;
      high = p.high;

      shouldTrain = p.shouldTrain;
      useRandomWeights = p.useRandomWeights;
      shouldSaveWeights = p.shouldSaveWeights;

      weightsFilePath = p.weightsFilePath;
   } // public void setConfigParams()

/**
 * Prints all configuration parameters to the terminal. 
 */
   public void echoConfigParams()
   {
      System.out.println(inputNodes + "-" + hiddenLayerNodes + "-" + outputNodes + " Network");
      System.out.println("Number of test cases: " + numberOfCases);
      if (shouldTrain)
      {
         System.out.println("Training");
         System.out.printf("Error threshold: %.4f\n", errorThreshold);
         System.out.println("Maximum iterations: " + maxIterations);
         System.out.println("Learning rate: " + lambda);
      }
      else
      {
         System.out.println("Running"); 
      }

      if (shouldSaveWeights)
      {
         System.out.println("Saving weights to file with path " + weightsFilePath);
      }

      if (useRandomWeights)
      {
         System.out.println("Random weight values in the range (" + low + ", " + high + ")");
      }
      else
      {
         System.out.println("Loading weights from file with path " + weightsFilePath);
      }
   } // public void echoConfigParams()

/**
 * Initializes and allocates memory for all arrays necessary for the computation of the network
 */
   public void allocateArrayMemory()
   {
      a = new double[inputNodes];
      h = new double[hiddenLayerNodes];
      F = new double[outputNodes];

      weightsKJ = new double[inputNodes][hiddenLayerNodes];
      weightsJI = new double[hiddenLayerNodes][outputNodes];

      testCases = new double[numberOfCases][inputNodes];
      trueOutputs = new double[numberOfCases][outputNodes]; 
      
      if (shouldTrain)
      {  
         upperThetaJ = new double[hiddenLayerNodes];
         lowerPsi = new double[outputNodes];

         changeInWeightsKJ = new double[inputNodes][hiddenLayerNodes];
         changeInWeightsJI = new double[hiddenLayerNodes][outputNodes];

         changeInErrorKJ = new double[inputNodes][hiddenLayerNodes];
         changeInErrorJI = new double[hiddenLayerNodes][outputNodes];
      } // if (train)
   } // public void allocateArrayMemory()

/**
 * The activation function (a sigmoid) used for the network. Helps caluclate the hidden layer and output layer activations.
 * 
 * @param x activations in previous layer * weights, or uppercase theta
 * @return  the value of the activation
 */
   public double f(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }

/**
* The derivative of the activation function, i.e the change in the activation function. Returns the slope of f at the given
* x value. 
* 
* @param x  activations in previous layer * weights, or uppercase theta
* @return  the change in the activation function
*/
   public double fPrime(double x)
   {
      double res = f(x);
      return res * (1.0 - res);
   }

/**
* Generates a random number within a range between low and high
*/
   public double randomize()
   {
      return Math.random() * (high - low) + low;
   }

/**
 * Creates all test cases and their respective answers manually and stores them in arrays as a truth table
 */
   public void createTruthTable()
   {
      int k, i, testCase; 

      for (testCase = 0; testCase < numberOfCases; testCase++)
      {
         for (k = 0; k < inputNodes; k++)
         {
            testCases[testCase][k] = p.testCases[testCase][k];
         }

         for (i = 0; i < outputNodes; i++)
         {
            trueOutputs[testCase][i] = p.trueOutputs[testCase][i];
         }
      } // for (testCase = 0; testCase < numberOfCases; testCase++)
   } // public void createTruthTable()

/**
 * Randomizes the weights for the KJ and JI layers in a range between low (the lower bound of the range) and 
 * high (the upper bound of the range)
*/
   public void randomizeWeights()
   {
      int k, j, i;

      for (k = 0; k < inputNodes; k++)
      {
         for (j = 0; j < hiddenLayerNodes; j++)
         {
            weightsKJ[k][j] = randomize();
         }
      }

      for (j = 0; j < hiddenLayerNodes; j++)
      {
         for (i = 0; i < outputNodes; i++)
         {
            weightsJI[j][i] = randomize();
         }
      }
   } // public void randomizeWeights()

/**
 * Sets the test cases and the expected outputs for each case, and randomizes the weights in the KJ and 
 * JI layers
 */
   public void populateArrays() throws IOException
   {
      createTruthTable();

      if (useRandomWeights)
      {
         randomizeWeights();
      }
      else
      {
         loadWeights();
      }
   } // public void populateArrays()

/**
 * Runs the network as it is; unlike train, run does not train the weights and instead immediately
 * calculates the output activation
 */
   public void run(int testCase)
   {
      int k, j, i;
      double theta;

      for (k = 0; k < inputNodes; k++)
      {
         a[k] = testCases[testCase][k];
      }

      for (j = 0; j < hiddenLayerNodes; j++)
      {
         theta = 0.0;

         for (k = 0; k < inputNodes; k++)
         {
            theta += a[k] * weightsKJ[k][j];
         }
         h[j] = f(theta);
      }

      for (i = 0; i < outputNodes; i++)
      {
         theta = 0.0;

         for (j = 0; j < hiddenLayerNodes; j++)
         {
            theta += h[j] * weightsJI[j][i];
         }
         F[i] = f(theta);
      }
   } // public void run()

/**
 * Runs for all of the test cases
 */
   public void runAllTestCases()
   {
      int testCase;

      for (testCase = 0; testCase < numberOfCases; testCase++)
      {
         run(testCase);
      }
   } // public void runAllTestCases()

/**
 * Calculates necessary values for training the network (including the activations, omega, psi, and theta values)
 */
   public void runForTrain(int testCase)
   {
      int k, j, i;
      double theta, omega;

      for (j = 0; j < hiddenLayerNodes; j++)
      {
         upperThetaJ[j] = 0.0;

         for (k = 0; k < inputNodes; k++)
         {
            upperThetaJ[j] += a[k] * weightsKJ[k][j];
         }
         h[j] = f(upperThetaJ[j]);
      }

      for (i = 0; i < outputNodes; i++)
      {
         theta = 0.0;

         for (j = 0; j < hiddenLayerNodes; j++)
         {
            theta += h[j] * weightsJI[j][i];
         }
         F[i] = f(theta);
         omega = trueOutputs[testCase][i] - F[i]; 
         lowerPsi[i] = omega * fPrime(theta);
      }
   } // public void runForTrain(int testCase)

/**
 * Runs backpropagation on the network, which trains in a more optimized way (less loops) that goes also 
 * goes backwards to help in training the network
 */
   public void backpropagation(int testCase)
   {
      int k, j, i;
      double omega, psi;

      for (j = 0; j < hiddenLayerNodes; j++)
      {
         omega = 0.0;
         for (i = 0 ; i < outputNodes; i++)
         {
            omega += lowerPsi[i] * weightsJI[j][i];
            weightsJI[j][i] += lambda * h[j] * lowerPsi[i];
         }

         psi = omega * fPrime(upperThetaJ[j]);

         for (k = 0; k < inputNodes; k++)
         {
            weightsKJ[k][j] += lambda * a[k] * psi;
         }
      }
   } // public void backpropagation(int testCase)

/**
 * Repeatedly iterates through, modifying the weights on each pass to better match the output activation to the 
 * expected output. Finishes training when the current error is under the error threshold, or when the maximum
 * number of iterations has been reached
 */
   public void train()
   {
      int k, i, testCase; 

      totalError = Double.MAX_VALUE;
      avgError = totalError / ((double) numberOfCases);

      System.out.println("avgError: " + avgError);

      while (avgError > errorThreshold && iterations < maxIterations)
      {
         totalError = 0.0;

         iterations++;

         for (testCase = 0; testCase < numberOfCases; testCase++)
         {
            for (k = 0; k < inputNodes; k++)
            {
               a[k] = testCases[testCase][k];
            }

            runForTrain(testCase);
            backpropagation(testCase);

            run(testCase);
            for (i = 0; i < outputNodes; i++)
            {
               totalError += 0.5 * (trueOutputs[testCase][i] - F[i]) * (trueOutputs[testCase][i] - F[i]);
            }
         } // for (testCase = 0; testCase < 4; testCase++)
         avgError = totalError / ((double) (numberOfCases));
      } // while (error > errorThreshold && iterations < maxIterations)
      totalError = 0.0;
   } // public void train()

/**
 * Prints the truth table, including all cases and their expected values. 
 */
   public void printTruthTable()
   {
      int testCase, k, i;

      System.out.println("\nTruth Table");

      for (k = 0; k < inputNodes; k++)
      {
         System.out.print("a" + k + "  ");
      }

      System.out.print("  ");

      for (i = 0; i < outputNodes; i++)
      {
         System.out.print("F" + i + "  ");
      }

      System.out.println();

      for (testCase = 0; testCase < numberOfCases; testCase++)
      {
         for (k = 0; k < inputNodes; k++)
         {
            System.out.print(testCases[testCase][k] + " ");
         }

         System.out.print("| ");

         for (i = 0; i < outputNodes; i++)
         {
            System.out.print(trueOutputs[testCase][i] + " ");
         }
         System.out.println();
      } // for (testCase = 0; testCase < numberOfCases; testCase++)
   } // public void printTruthTable()

/*
* Accept a value representing seconds elapsed and print out a decimal value in easier to digest units
* Code taken from Dr. Nelson on Schoology. 
*/
   public void printTime(double seconds)
   {
      double minutes, hours, days, weeks;

      System.out.printf("Elapsed time: ");

      if (seconds < 1.)
         System.out.printf("%g milliseconds", seconds * UNIT_CONVERSION_THOUSAND);

      else if (seconds < SECONDS_IN_MINUTE)
         System.out.printf("%g seconds", seconds);

      else
      {
         minutes = seconds / SECONDS_IN_MINUTE;

         if (minutes < MINUTES_IN_HOUR)
            System.out.printf("%g minutes", minutes);

         else
         {
            hours = minutes / MINUTES_IN_HOUR;

            if (hours < HOURS_IN_DAY)
               System.out.printf("%g hours", hours);

            else
            {
               days = hours / HOURS_IN_DAY;

               if (days < 7.)
                  System.out.printf("%g days", days);

               else
               {
                  weeks = days / DAYS_IN_WEEK;
                  System.out.printf("%g weeks", weeks);
               }
            } // if (hours < 24.)...else
         } // if (minutes < 60.)...else
      } // else if (seconds < 60.)...else
      System.out.printf("\n\n");
   } // public void printTime(double seconds)

/**
 * reportResults prints the truth table (if training, it also prints the actual output activation produced for 
 * each test case), the overall error across all test cases and the number of iterations taken. In addition, prints
 * the reason for exiting (i.e, maximum iterations was reached, or the total error is under the error threshold)
 * 
 * @precondition  only called after either run() or train() is called 
 */
   public void reportResults()
   {
      int testCase, i;

      printTruthTable();

      System.out.println("\nOutputs");

      for (testCase = 0; testCase < numberOfCases; testCase++)
      {
         System.out.println();
         run(testCase);
         for (i = 0; i < outputNodes; i++)
         {
            System.out.printf("%.17f ", F[i]);
            totalError += 0.5 * (trueOutputs[testCase][i] - F[i]) * (trueOutputs[testCase][i] - F[i]);
         }
      } // for (testCase = 0; testCase < numberOfCases; testCase++)

      System.out.println();
      
      if (shouldTrain)
      {
         System.out.println("\n\nnumber of iterations: " + iterations);

         System.out.println("Reason(s) for exiting: "); 
         
         if (iterations >= maxIterations)
         {
            System.out.println("Max iterations (" + maxIterations + ") reached");
         }

         if (avgError < errorThreshold)
         {
            System.out.printf("error under error threshold (%.4f)\n", errorThreshold);
         }

         System.out.printf("average error: %.4f\n", avgError);
      } // if (shouldTrain)
      else
      {
         avgError = totalError / ((double) (numberOfCases));

         System.out.printf("average error: %.4f\n", avgError);
      }
   } // public void reportResults()

/**
 * Saves the calculated weights to a file for later usage
 * 
 * @throws IOException may be thrown when writing to the file fails 
 */
   public void saveWeights() throws IOException
   {
      int k, j, i;

      out = new PrintWriter(new FileWriter(weightsFilePath), true);
      
      for (k = 0; k < inputNodes; k++)
      {
         for (j = 0; j < hiddenLayerNodes; j++)
         {
            out.println(weightsKJ[k][j]);
         }
      }

      for (j = 0; j < hiddenLayerNodes; j++)
      {
         for (i = 0; i < outputNodes; i++)
         {
            out.println(weightsJI[j][i]);
         }
      }
      out.close();
   } // public void saveWeights() throws IOException

/**
 * Loads the weights from a file into the current weights arrays
 * 
 * @throws IOException may be thrown when loading from the file fails
 */
   public void loadWeights() throws IOException
   {
      int k, j, i;

      in = new Scanner(new File(weightsFilePath));

      for (k = 0; k < inputNodes; k++)
      {
         for (j = 0; j < hiddenLayerNodes; j++)
         {
            weightsKJ[k][j] = in.nextDouble();
         }
      }

      for (j = 0; j < hiddenLayerNodes; j++)
      {
         for (i = 0; i < outputNodes; i++)
         {
            weightsJI[j][i] = in.nextDouble();
         }
      }

      in.close();
   } // public void loadWeights() throws IOException

/**
 * The tester for the ABC Backpropagation Network
 * 
 * @param args the parameters for the main method
 * @throws IOException 
 */
   public static void main(String args[]) throws IOException
   {
      double startTime; 

      startTime = System.nanoTime();

      ABCNetwork n = new ABCNetwork();

      try
      {
         n.parseConfigParams(args[0]);
      }
      catch (ArrayIndexOutOfBoundsException e)
      {
         System.out.println("No config filename passed, using default file instead");
         n.parseConfigParams(DEFAULT_FILE_NAME);
      }

      n.setConfigParams();
      n.echoConfigParams();
      n.allocateArrayMemory();
      n.populateArrays();
      if (n.shouldTrain)
      {
         n.train();
      }
      else
      {
         n.runAllTestCases();
      }

      if (n.shouldSaveWeights)
      {
         n.saveWeights();
      }
      n.reportResults();

      n.printTime((System.nanoTime() - startTime) / 1000000000);
   } // public static void main(String args[]) throws IOException
} // public class AB1Network