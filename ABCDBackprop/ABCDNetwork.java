import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Scanner;

/**
 * The A-B-C-D Network is a multi-layer perceptron that uses gradient descent to minimize the error across all test cases. 
 * With every iteration, it takes a 'step downhill' the error curve and optimizes its existing weights to reduce the error 
 * as much as possible for every iteration. The network is based off a feed-forward model and uses backpropgation to optimize
 * calculations.
 * 
 * Table of Contents
 *  • public void parseConfigParams(String configFilePath)
 *  • public void setConfigParams()
 *  • public void echoConfigParams()
 *  • public void allocateArrayMemory()
 *  • public double f(double x)
 *  • public double fPrime(double x)
 *  • public double randomize()
 *  • public void createTruthTable()
 *  • public void randomizeWeights()
 *  • public void populateArrays()
 *  • public void run()
 *  • public void runAllTestCases()
 *  • public void runForTrain(int testCase)
 *  • public void backpropagation(int testCase)
 *  • public void train()
 *  • public void printTruthTable()
 *  • public void printTime(double seconds)
 *  • public void reportResults()
 *  • public void saveWeights()
 *  • public void loadWeights()
 * 
 * @author  Kyle Li
 * @version 22 April 2024
 * Date of creation: 1 February 2024
 */
public class ABCDNetwork
{
   static final int UNIT_CONVERSION_THOUSAND = 1000;
   static final int SECONDS_IN_MINUTE = 60;
   static final int MINUTES_IN_HOUR = 60;
   static final int HOURS_IN_DAY = 24;
   static final int DAYS_IN_WEEK = 7; 
   static final String DEFAULT_FILE_NAME = "./default.txt";

   static final int INPUTLAYER = 0;
   static final int HIDLAYER1 = 1;
   static final int HIDLAYER2 = 2;
   static final int OUTPUTLAYER = 3; 

   static final int NUM_OF_LAYERS = 4;

   int inputNodes;
   int hiddenLayerNodes1;
   int hiddenLayerNodes2;
   int outputNodes;

   double activations[][];
   
   double theta[][];
   double psi[][]; 

   double totalError;
   double avgError; 
   double errorThreshold; 

   double weights[][][]; 

   double lambda; 

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

   ActivationFunction act;
   NNParser p; 

   Scanner in;
   PrintWriter out;

/**
 * Parses the given configuration file and gets the values for all the variables in it
 * 
 * @param configFilePath   the filepath to the configuration file
 * @throws FileNotFoundException 
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
      hiddenLayerNodes1 = p.hiddenLayerNodes1;
      hiddenLayerNodes2 = p.hiddenLayerNodes2;
      outputNodes = p.outputNodes;

      errorThreshold = p.errorThreshold;

      maxIterations = p.maxIterations;

      numberOfCases = p.numberOfCases;

      lambda = p.lambda;

      low = p.low;
      high = p.high;

      act = p.act;

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
      System.out.println(inputNodes + "-" + hiddenLayerNodes1 + "-" + hiddenLayerNodes2 + "-" + outputNodes + " Network");
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
      activations = new double[NUM_OF_LAYERS][];
      activations[INPUTLAYER] = new double[inputNodes];
      activations[HIDLAYER1] = new double[hiddenLayerNodes1];
      activations[HIDLAYER2] = new double[hiddenLayerNodes2];
      activations[OUTPUTLAYER] = new double[outputNodes];

      weights = new double[NUM_OF_LAYERS][][];
      weights[INPUTLAYER] = new double[inputNodes][hiddenLayerNodes1];
      weights[HIDLAYER1] = new double[hiddenLayerNodes1][hiddenLayerNodes2];
      weights[HIDLAYER2] = new double[hiddenLayerNodes2][outputNodes];

      testCases = new double[numberOfCases][inputNodes];
      trueOutputs = new double[numberOfCases][outputNodes]; 
      
      if (shouldTrain)
      {  
         theta = new double[NUM_OF_LAYERS][];
         theta[INPUTLAYER] = new double[inputNodes];
         theta[HIDLAYER1] = new double[hiddenLayerNodes1];
         theta[HIDLAYER2] = new double[hiddenLayerNodes2];
         theta[OUTPUTLAYER] = new double[outputNodes];

         psi = new double[NUM_OF_LAYERS][];
         psi[INPUTLAYER] = new double[inputNodes];
         psi[HIDLAYER1] = new double[hiddenLayerNodes1];
         psi[HIDLAYER2] = new double[hiddenLayerNodes2];
         psi[OUTPUTLAYER] = new double[outputNodes];
      } // if (shouldTrain)
   } // public void allocateArrayMemory()

/**
 * The activation function (a sigmoid) used for the network. Helps caluclate the hidden layer and output layer activations.
 * 
 * @param x activations in previous layer * weights, or uppercase theta
 * @return  the value of the activation
 */
   public double f(double x)
   {
      return act.f(x);
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
      return act.fPrime(x);
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
      int m, i, testCase; 

      for (testCase = 0; testCase < numberOfCases; testCase++)
      {
         for (m = 0; m < inputNodes; m++)
         {
            testCases[testCase][m] = p.testCases[testCase][m];
         }

         for (i = 0; i < outputNodes; i++)
         {
            trueOutputs[testCase][i] = p.trueOutputs[testCase][i];
         }
      } // for (testCase = 0; testCase < numberOfCases; testCase++)
   } // public void createTruthTable()

/**
 * Randomizes the weights for the layers in a range between low (the lower bound of the range) and 
 * high (the upper bound of the range)
*/
   public void randomizeWeights()
   {
      int m, k, j, i, n;

      n = INPUTLAYER;
      for (m = 0; m < inputNodes; m++)
      {
         for (k = 0; k < hiddenLayerNodes1; k++)
         {
            weights[n][m][k] = randomize();
         }
      }

      n = HIDLAYER1;
      for (k = 0; k < hiddenLayerNodes1; k++)
      {
         for (j = 0; j < hiddenLayerNodes2; j++)
         {
            weights[n][k][j] = randomize();
         }
      }

      n = HIDLAYER2; 
      for (j = 0; j < hiddenLayerNodes2; j++)
      {
         for (i = 0; i < outputNodes; i++)
         {
            weights[n][j][i] = randomize();
         }
      }
   } // public void randomizeWeights()

/**
 * Sets the test cases and the expected outputs for each case, and randomizes the weights in the layers
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
   public void run()
   {
      int m, k, j, i, n;
      double tempTheta;

      n = INPUTLAYER;
      for (k = 0; k < hiddenLayerNodes1; k++)
      {
         tempTheta = 0.0;

         for (m = 0; m < inputNodes; m++)
         {
            tempTheta += activations[n][m] * weights[n][m][k];
         }
         activations[n + 1][k] = f(tempTheta);
      } // for (k = 0; k < hiddenLayerNodes1; k++)

      n = HIDLAYER1;
      for (j = 0; j < hiddenLayerNodes2; j++)
      {
         tempTheta = 0.0;

         for (k = 0; k < hiddenLayerNodes1; k++)
         {
            tempTheta += activations[n][k] * weights[n][k][j];
         }
         activations[n + 1][j] = f(tempTheta);
      } // for (j = 0; j < hiddenLayerNodes2; j++)

      n = HIDLAYER2; 
      for (i = 0; i < outputNodes; i++)
      {
         tempTheta = 0.0;

         for (j = 0; j < hiddenLayerNodes2; j++)
         {
            tempTheta += activations[n][j] * weights[n][j][i];
         }
         activations[n + 1][i] = f(tempTheta);
      } // for (i = 0; i < outputNodes; i++)
   } // public void run()

/**
 * Runs for all of the test cases
 */
   public void runAllTestCases()
   {
      int testCase, m, n;

      n = INPUTLAYER;
      for (testCase = 0; testCase < numberOfCases; testCase++)
      {
         for (m = 0; m < inputNodes; m++)
         {
            activations[n][m] = testCases[testCase][m];
         }
         run();
      }
   } // public void runAllTestCases()

/**
 * Calculates necessary values for training the network (including the activations, omega, psi, and theta values)
 */
   public void runForTrain(int testCase)
   {
      int m, k, j, i, n;
      double omega;

      n = HIDLAYER1;
      for (k = 0; k < hiddenLayerNodes1; k++)
      {
         theta[n][k] = 0;
         for (m = 0; m < inputNodes; m++)
         {
            theta[n][k] += activations[n - 1][m] * weights[n - 1][m][k];
         }
         activations[n][k] = f(theta[n][k]);
      }

      n = HIDLAYER2;
      for (j = 0; j < hiddenLayerNodes2; j++)
      {
         theta[n][j] = 0;
         for (k = 0; k < hiddenLayerNodes1; k++)
         {
            theta[n][j] += activations[n - 1][k] * weights[n - 1][k][j];
         }
         activations[n][j] = f(theta[n][j]);
      }

      n = OUTPUTLAYER;
      for (i = 0; i < outputNodes; i++)
      {
         theta[n][i] = 0;
         for (j = 0; j < hiddenLayerNodes2; j++)
         {
            theta[n][i] += activations[n - 1][j] * weights[n - 1][j][i];
         }
         activations[n][i] = f(theta[n][i]);
         omega = trueOutputs[testCase][i] - activations[n][i];
         psi[n][i] = omega * fPrime(theta[n][i]);
      }
   } // public void runForTrain(int testCase)

/**
 * Runs backpropagation on the network, which trains in a more optimized way (less loops) that goes also 
 * goes backwards to help in training the network
 */
   public void backpropagation(int testCase)
   {
      int m, k, j, i, n;
      double omega;

      n = HIDLAYER2; 
      for (j = 0; j < hiddenLayerNodes2; j++)
      {
         omega = 0.0;
         for (i = 0; i < outputNodes; i++)
         {
            omega += psi[n + 1][i] * weights[n][j][i];
            weights[n][j][i] += lambda * activations[n][j] * psi[n + 1][i];
         }

         psi[n][j] = omega * fPrime(theta[n][j]);
      }

      n = HIDLAYER1;
      for (k = 0; k < hiddenLayerNodes1; k++)
      {
         omega = 0.0;
         for (j = 0; j < hiddenLayerNodes2; j++)
         {
            omega += psi[n + 1][j] * weights[n][k][j];
            weights[n][k][j] += lambda * activations[n][k] * psi[n + 1][j];
         }

         psi[n][k] = omega * fPrime(theta[n][k]);

         for (m = 0; m < inputNodes; m++)
         {
            weights[n - 1][m][k] += lambda * activations[n - 1][m] * psi[n][k]; 
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
      int m, i, n, testCase; 

      totalError = Double.MAX_VALUE;
      avgError = totalError / ((double) numberOfCases);

      System.out.println("avgError: " + avgError);

      while (avgError > errorThreshold && iterations < maxIterations)
      {
         totalError = 0.0;

         iterations++;

         for (testCase = 0; testCase < numberOfCases; testCase++)
         {
            n = INPUTLAYER;
            for (m = 0; m < inputNodes; m++)
            {
               activations[n][m] = testCases[testCase][m];
            }

            runForTrain(testCase);
            backpropagation(testCase);

            run();

            n = OUTPUTLAYER;
            for (i = 0; i < outputNodes; i++)
            {
               totalError += 0.5 * (trueOutputs[testCase][i] - activations[n][i]) * (trueOutputs[testCase][i] - activations[n][i]);
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
      int testCase, m, i;

      System.out.println("\nTruth Table");

      for (m = 0; m < inputNodes; m++)
      {
         System.out.print("a" + m + "  ");
      }

      System.out.print("  ");

      for (i = 0; i < outputNodes; i++)
      {
         System.out.print("F" + i + "  ");
      }

      System.out.println();

      for (testCase = 0; testCase < numberOfCases; testCase++)
      {
         for (m = 0; m < inputNodes; m++)
         {
            System.out.print(testCases[testCase][m] + " ");
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
      int testCase, m, i, n;

      printTruthTable();

      System.out.println("\nOutputs");

      for (testCase = 0; testCase < numberOfCases; testCase++)
      {
         System.out.println();

         n = INPUTLAYER;
         for (m = 0; m < inputNodes; m++)
         {
            activations[n][m] = testCases[testCase][m];
         }

         run();

         n = OUTPUTLAYER;
         for (i = 0; i < outputNodes; i++)
         {
            System.out.printf("%.17f ", activations[n][i]);
            totalError += 0.5 * (trueOutputs[testCase][i] - activations[n][i]) * (trueOutputs[testCase][i] - activations[n][i]);
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
      int m, k, j, i, n;

      out = new PrintWriter(new FileWriter(weightsFilePath), true);

      n = INPUTLAYER;
      for (m = 0; m < inputNodes; m++)
      {
         for (k = 0; k < hiddenLayerNodes1; k++)
         {
            out.println(weights[n][m][k]);
         }
      }
      
      n = HIDLAYER1;
      for (k = 0; k < hiddenLayerNodes1; k++)
      {
         for (j = 0; j < hiddenLayerNodes2; j++)
         {
            out.println(weights[n][k][j]);
         }
      }

      n = HIDLAYER2; 
      for (j = 0; j < hiddenLayerNodes2; j++)
      {
         for (i = 0; i < outputNodes; i++)
         {
            out.println(weights[n][j][i]);
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
      int m, k, j, i, n;

      in = new Scanner(new File(weightsFilePath));

      n = INPUTLAYER;
      for (m = 0; m < inputNodes; m++)
      {
         for (k = 0; k < hiddenLayerNodes1; k++)
         {
            weights[n][m][k] = in.nextDouble();
         }
      }

      n = HIDLAYER1;
      for (k = 0; k < hiddenLayerNodes1; k++)
      {
         for (j = 0; j < hiddenLayerNodes2; j++)
         {
            weights[n][k][j] = in.nextDouble();
         }
      }

      n = HIDLAYER2;
      for (j = 0; j < hiddenLayerNodes2; j++)
      {
         for (i = 0; i < outputNodes; i++)
         {
            weights[n][j][i] = in.nextDouble();
         }
      }

      in.close();
   } // public void loadWeights() throws IOException

/**
 * The tester for the ABCD Backpropagation Network
 * 
 * @param args the parameters for the main method
 * @throws IOException 
 */
   public static void main(String args[]) throws IOException 
   {
      double startTime; 

      startTime = System.nanoTime();

      ABCDNetwork n = new ABCDNetwork();

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