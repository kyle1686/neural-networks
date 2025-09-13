import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Scanner;

/**
 * The N-Layer Network is a multi-layer perceptron that uses gradient descent to minimize the error across all test cases. 
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
 * @version 30 April 2024
 * Date of creation: 1 February 2024
 */
public class NLayer
{
   static final int UNIT_CONVERSION_THOUSAND = 1000;
   static final int SECONDS_IN_MINUTE = 60;
   static final int MINUTES_IN_HOUR = 60;
   static final int HOURS_IN_DAY = 24;
   static final int DAYS_IN_WEEK = 7; 
   static final String DEFAULT_FILE_NAME = "./default.txt";

   static final int SHOULD_KEEP_ALIVE = 1; 
   static final int NO_REMAINDER = 0;

   static final int INPUTLAYER = 0; 
   static final int HIDLAYER1 = 1;

   int layers[]; 

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

   int keepAlive;

   String weightsFilePath;

   int n;

   ActivationFunction act;
   NNParser p; 

   Scanner fin;
   PrintWriter fout;

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
      n = p.n; 

      layers = p.layers; 

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
      keepAlive = p.keepAlive;

      weightsFilePath = p.weightsFilePath;
   } // public void setConfigParams()

/**
 * Prints all configuration parameters to the terminal. 
 */
   public void echoConfigParams()
   {
      int curr; 

      for (curr = 0; curr < n - 1; curr++)
      {
         System.out.print(layers[curr] + "-"); 
      }
      System.out.println(layers[n - 1] + " Network");

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
      int index; 

      activations = new double[n][];
      for (index = 0; index < n; index++)
      {
         activations[index] = new double[layers[index]];
      }

      weights = new double[n][][];
      for (index = 0; index < n - 1; index++)
      {
         weights[index] = new double[layers[index]][layers[index + 1]];
      }

      testCases = new double[numberOfCases][layers[INPUTLAYER]];
      trueOutputs = new double[numberOfCases][layers[n - 1]]; 
      
      if (shouldTrain)
      {  
         theta = new double[n][];
         for (index = 0; index < n; index++)
         {
            theta[index] = new double[layers[index]];
         }

         psi = new double[n][];
         for (index = 0; index < n; index++)
         {
            psi[index] = new double[layers[index]];
         }
      } // if (shouldTrain)
   } // public void allocateArrayMemory()

/**
 * The activation function (a sigmoid) used for the network. Helps calculate the hidden layer and output layer activations.
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
         for (m = 0; m < layers[INPUTLAYER]; m++)
         {
            testCases[testCase][m] = p.testCases[testCase][m];
         }

         for (i = 0; i < layers[n - 1]; i++)
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
      int alpha, beta, gamma;

      for (alpha = 1; alpha < n; alpha++)
      {
         for (beta = 0; beta < layers[alpha]; beta++)
         {
            for (gamma = 0; gamma < layers[alpha - 1]; gamma++)
            {
               weights[alpha - 1][gamma][beta] = randomize();
            }
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
      int alpha, beta, gamma;
      double tempTheta;

      for (alpha = 1; alpha < n; alpha++)
      {
         for (beta = 0; beta < layers[alpha]; beta++)
         {
            tempTheta = 0.0;

            for (gamma = 0; gamma < layers[alpha - 1]; gamma++)
            {
               tempTheta += activations[alpha - 1][gamma] * weights[alpha - 1][gamma][beta];
            }
            activations[alpha][beta] = f(tempTheta);
         } // for (beta = 0; beta < layers[alpha]; beta++)
      } // for (alpha = 1; alpha < n; alpha++)
   } // public void run()

/**
 * Runs for all of the test cases
 */
   public void runAllTestCases()
   {
      int testCase, inp;

      for (testCase = 0; testCase < numberOfCases; testCase++)
      {
         for (inp = 0; inp < layers[INPUTLAYER]; inp++)
         {
            activations[INPUTLAYER][inp] = testCases[testCase][inp];
         }
         run();
      }
   } // public void runAllTestCases()

/**
 * Calculates necessary values for training the network (including the activations, omega, psi, and theta values)
 */
   public void runForTrain(int testCase)
   {
      int alpha, beta, gamma;

      for (alpha = 1; alpha < n - 1; alpha++)
      {
         for (beta = 0; beta < layers[alpha]; beta++)
         {
            theta[alpha][beta] = 0.0;

            for (gamma = 0; gamma < layers[alpha - 1]; gamma++)
            {
               theta[alpha][beta] += activations[alpha - 1][gamma] * weights[alpha - 1][gamma][beta];
            }
            activations[alpha][beta] = f(theta[alpha][beta]);
         } // for (beta = 0; beta < layers[alpha]; beta++)
      } // for (alpha = 1; alpha < n - 1; alpha++)

      alpha = n - 1;
      for (beta = 0; beta < layers[alpha]; beta++)
      {
         theta[alpha][beta] = 0.0;
         
         for (gamma = 0; gamma < layers[alpha - 1]; gamma++)
         {
            theta[alpha][beta] += activations[alpha - 1][gamma] * weights[alpha - 1][gamma][beta];
         }
         activations[alpha][beta] = f(theta[alpha][beta]);
         psi[alpha][beta] = (trueOutputs[testCase][beta] - activations[alpha][beta]) * fPrime(theta[alpha][beta]);
      } // for (beta = 0; beta < layers[alpha]; beta++)
   } // public void runForTrain(int testCase)

/**
 * Runs backpropagation on the network, which trains in a more optimized way (less loops) that goes also 
 * goes backwards to help in training the network
 */
   public void backpropagation(int testCase)
   {
      int alpha, beta, gamma, inp;
      double omega;

      for (alpha = n - 2; alpha > 1; alpha--)
      {
         for (gamma = 0; gamma < layers[alpha]; gamma++)
         {  
            omega = 0.0;
            for (beta = 0; beta < layers[alpha + 1]; beta++)
            {
               omega += psi[alpha + 1][beta] * weights[alpha][gamma][beta];
               weights[alpha][gamma][beta] += lambda * activations[alpha][gamma] * psi[alpha + 1][beta];
            }
            psi[alpha][gamma] = omega * fPrime(theta[alpha][gamma]);
         } // for (gamma = 0; gamma < layers[alpha]; gamma++)
      } // for (alpha = n - 2; alpha > 1; alpha--)

      alpha = HIDLAYER1;
      for (gamma = 0; gamma < layers[alpha]; gamma++)
      {
         omega = 0.0;
         for (beta = 0; beta < layers[alpha + 1]; beta++)
         {
            omega += psi[alpha + 1][beta] * weights[alpha][gamma][beta];
            weights[alpha][gamma][beta] += lambda * activations[alpha][gamma] * psi[alpha + 1][beta];
         }

         for (inp = 0; inp < layers[INPUTLAYER]; inp++)
         {
            weights[alpha - 1][inp][gamma] += lambda * activations[alpha - 1][inp] * omega * fPrime(theta[alpha][gamma]);
         }
      } // for (gamma = 0; gamma < layers[alpha]; gamma++)
   } // public void backpropagation(int testCase)

/**
 * Repeatedly iterates through, modifying the weights on each pass to better match the output activation to the 
 * expected output. Finishes training when the current error is under the error threshold, or when the maximum
 * number of iterations has been reached
 */
   public void train()
   {
      int inp, out, testCase; 

      totalError = Double.MAX_VALUE;
      avgError = totalError / ((double) numberOfCases);

      System.out.println("avgError: " + avgError + "\n");

      while (avgError > errorThreshold && iterations < maxIterations)
      {
         totalError = 0.0;

         iterations++;

         for (testCase = 0; testCase < numberOfCases; testCase++)
         {
            for (inp = 0; inp < layers[INPUTLAYER]; inp++)
            {
               activations[INPUTLAYER][inp] = testCases[testCase][inp];
            }

            runForTrain(testCase);
            backpropagation(testCase);

            run();

            for (out = 0; out < layers[n - 1]; out++)
            {
               totalError += 0.5 * (trueOutputs[testCase][out] - activations[n - 1][out]) * (trueOutputs[testCase][out] - activations[n - 1][out]);
            }
         } // for (testCase = 0; testCase < 4; testCase++)
         avgError = totalError / ((double) (numberOfCases));

         if (keepAlive >= SHOULD_KEEP_ALIVE && iterations % keepAlive == NO_REMAINDER) 
            System.out.printf("Iteration %d, Error = %f\n", iterations, avgError);
      } // while (error > errorThreshold && iterations < maxIterations)
      totalError = 0.0;
   } // public void train()

/**
 * Prints the truth table, including all cases and their expected values. 
 */
   public void printTruthTable()
   {
      int testCase, inp, out;

      System.out.println("\nTruth Table");

      for (inp = 0; inp < layers[INPUTLAYER]; inp++)
      {
         System.out.print("a" + inp + "  ");
      }

      System.out.print("  ");

      for (out = 0; out < layers[n - 1]; out++)
      {
         System.out.print("F" + out + "  ");
      }

      System.out.println();

      for (testCase = 0; testCase < numberOfCases; testCase++)
      {
         for (inp = 0; inp < layers[INPUTLAYER]; inp++)
         {
            System.out.print(testCases[testCase][inp] + " ");
         }

         System.out.print("| ");

         for (out = 0; out < layers[n - 1]; out++)
         {
            System.out.print(trueOutputs[testCase][out] + " ");
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
      int testCase, inp, out; 

      printTruthTable();

      System.out.println("\nOutputs");

      for (testCase = 0; testCase < numberOfCases; testCase++)
      {
         System.out.println();

         for (inp = 0; inp < layers[INPUTLAYER]; inp++)
         {
            activations[INPUTLAYER][inp] = testCases[testCase][inp];
         }

         run();

         for (out = 0; out < layers[n - 1]; out++)
         {
            System.out.printf("%.17f ", activations[n - 1][out]);
            totalError += 0.5 * (trueOutputs[testCase][out] - activations[n - 1][out]) * (trueOutputs[testCase][out] - activations[n - 1][out]);
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
      int alpha, beta, gamma;

      fout = new PrintWriter(new FileWriter(weightsFilePath), true);

      for (alpha = 1; alpha < n; alpha++)
      {
         for (gamma = 0; gamma < layers[alpha - 1]; gamma++)
         {
            for (beta = 0; beta < layers[alpha]; beta++)
            {
               fout.println(weights[alpha - 1][gamma][beta]);
            }
         }
      }
      fout.close();
   } // public void saveWeights() throws IOException

/**
 * Loads the weights from a file into the current weights arrays
 * 
 * @throws IOException may be thrown when loading from the file fails
 */
   public void loadWeights() throws IOException
   {
      int alpha, beta, gamma;

      fin = new Scanner(new File(weightsFilePath));

      for (alpha = 1; alpha < n; alpha++)
      {
         for (gamma = 0; gamma < layers[alpha - 1]; gamma++)
         {
            for (beta = 0; beta < layers[alpha]; beta++)
            {
               weights[alpha - 1][gamma][beta] = fin.nextDouble(); 
            }
         }
      }

      fin.close();
   } // public void loadWeights() throws IOException

/**
 * The tester for the N-Layer Network
 * 
 * @param args the parameters for the main method
 * @throws IOException 
 */
   public static void main(String args[]) throws IOException 
   {
      double startTime; 

      startTime = System.nanoTime();

      NLayer n = new NLayer();

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