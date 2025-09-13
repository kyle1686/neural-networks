import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Scanner;

/**
 * The A-B-C Network is a multi-layer perceptron that uses gradient descent to minimize the error across all test cases. 
 * With every iteration, it takes a 'step downhill' the error curve and optimizes its existing weights to reduce the error 
 * as much as possible for every iteration. The network is based off a feed-forward model. 
 * 
 * @author  Kyle Li
 * @version 4 March 2024
 * Date of creation: 1 February 2024
 */
public class ABCNetwork
{
   final int INPUT_NODES = 2;
   final int HIDDEN_LAYER_NODES = 5; 
   final int OUTPUT_NODES = 3;

   final double ERROR_THRESHOLD = 0.0002;
   
   final int MAX_ITERATIONS = 100000;

   final double LEARNING_RATE = 0.3;

   final int NUMBER_OF_CASES = 4;

   final double LOWER_BOUND = 0.1;
   final double UPPER_BOUND = 1.5;

   int inputNodes;
   int hiddenLayerNodes;
   int outputNodes;

   double a[]; 
   double h[]; 
   double F[]; 
   
   double upperThetaJ[];
   double upperThetaI[];
   double upperOmega[];
   double lowerOmega[];
   double upperPsi[];
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
   double actualOutputs[][];

   double low;
   double high;

   boolean useRandomWeights;
   boolean shouldTrain;
   boolean shouldSaveWeights;

   String filePath;

   Scanner in;
   PrintWriter out;

/**
 * Sets configuration parameters, initializing variables to their respective constant values.
 */
   public void setConfigParams()
   {
      inputNodes = INPUT_NODES;
      hiddenLayerNodes = HIDDEN_LAYER_NODES;
      outputNodes = OUTPUT_NODES;

      errorThreshold = ERROR_THRESHOLD;

      maxIterations = MAX_ITERATIONS;

      numberOfCases = NUMBER_OF_CASES;

      lambda = LEARNING_RATE;

      low = LOWER_BOUND;
      high = UPPER_BOUND;

      shouldTrain = true;
      useRandomWeights = true;
      shouldSaveWeights = true;

      filePath = "./weights.txt";
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
      System.out.println("Random weight values in the range (" + low + ", " + high + ")");
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
      actualOutputs = new double[numberOfCases][outputNodes];
      
      if (shouldTrain)
      {  
         upperThetaJ = new double[hiddenLayerNodes];
         upperThetaI = new double[outputNodes];

         upperOmega = new double[hiddenLayerNodes];
         lowerOmega = new double[outputNodes];
         upperPsi = new double[hiddenLayerNodes];
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
      testCases[0][0] = 0.0;
      testCases[0][1] = 0.0;
      trueOutputs[0][0] = 0.0;
      trueOutputs[0][1] = 0.0;
      trueOutputs[0][2] = 0.0;

      testCases[1][0] = 0.0;
      testCases[1][1] = 1.0;
      trueOutputs[1][0] = 0.0;
      trueOutputs[1][1] = 1.0;
      trueOutputs[1][2] = 1.0;

      testCases[2][0] = 1.0;
      testCases[2][1] = 0.0;
      trueOutputs[2][0] = 0.0;
      trueOutputs[2][1] = 1.0;
      trueOutputs[2][2] = 1.0;

      testCases[3][0] = 1.0;
      testCases[3][1] = 1.0;
      trueOutputs[3][0] = 1.0;
      trueOutputs[3][1] = 1.0;
      trueOutputs[3][2] = 0.0;
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
 * Calculates the value of the hidden activation(s) and the final output activation when running
 */
   public void calculateActivationsForRun()
   {
      int k, j, i;
      double theta;

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
   } // public void calculateActivationsForRun()

/**
 * Runs the network as it is on all test cases; unlike train, run does not optimize the weights and immediately
 * calculates the output activation
 */
   public void run()
   {
      int testCase, k, i;

      for (testCase = 0; testCase < numberOfCases; testCase++)
      {
         for (k = 0; k < inputNodes; k++)
         {
            a[k] = testCases[testCase][k];
         }
            
         calculateActivationsForRun();

         for (i = 0; i < outputNodes; i++)
         {
            actualOutputs[testCase][i] = F[i];
         }

         for (i = 0; i < outputNodes; i++)
         {
            totalError += 0.5 * (trueOutputs[testCase][i] - F[i]) * (trueOutputs[testCase][i] - F[i]);
         }
      } // for (testCase = 0; testCase < numberOfCases; testCase++)
      avgError = totalError / (double) (numberOfCases);
   } // public void run()

/**
 * Calculates the change in error, the change in weights, and updates the actual weights in both the KJ 
 * layer and the JI layer between a range. 
 */
   public void updateWeights()
   {
      int k, j, i;

      for (k = 0; k < inputNodes; k++)
      {
         for (j = 0; j < hiddenLayerNodes; j++)
         {
            changeInErrorKJ[k][j] = -a[k] * upperPsi[j];
            changeInWeightsKJ[k][j] = -lambda * changeInErrorKJ[k][j];
         }
      }

      for (j = 0; j < hiddenLayerNodes; j++)
      {
         for (i = 0; i < outputNodes; i++)
         {
            changeInErrorJI[j][i] = -h[j] * lowerPsi[i];
            changeInWeightsJI[j][i] = -lambda * changeInErrorJI[j][i];
         }
      }
      
      for (k = 0; k < inputNodes; k++)
      {
         for (j = 0; j < hiddenLayerNodes; j++)
         {
            weightsKJ[k][j] += changeInWeightsKJ[k][j];
         }
      }

      for (j = 0; j < hiddenLayerNodes; j++)
      {
         for (i = 0; i < outputNodes; i++)
         {
            weightsJI[j][i] += changeInWeightsJI[j][i]; 
         }
      }
   } // public void updateWeights()

/**
 * Calculates the value of the hidden activation(s) and the final output activation when training
 */
   public void calculateActivationsForTrain()
   {
      int k, j, i;

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
         upperThetaI[i] = 0.0;

         for (j = 0; j < hiddenLayerNodes; j++)
         {
            upperThetaI[i] += h[j] * weightsJI[j][i];
         }
         F[i] = f(upperThetaI[i]);
      }
   } // public void calculateActivationsForTrain()

/**
 * Repeatedly iterates through, modifying the weights on each pass to better match the output activation to the 
 * expected output. Finishes training when the current error is under the error threshold, or when the maximum
 * number of iterations has been reached
 */
   public void train()
   {
      int k, j, i, testCase; 

      totalError = Double.MAX_VALUE;
      avgError = totalError / (double) numberOfCases;

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

            calculateActivationsForTrain();

            for (i = 0; i < outputNodes; i++)
            {
               lowerOmega[i] = trueOutputs[testCase][i] - F[i]; 
               lowerPsi[i] = lowerOmega[i] * fPrime(upperThetaI[i]);
            }

            for (j = 0; j < hiddenLayerNodes; j++)
            {
               upperOmega[j] = 0.0;
               for (i = 0; i < outputNodes; i++)
               {
                  upperOmega[j] += lowerPsi[i] * weightsJI[j][i];
               }
               upperPsi[j] = upperOmega[j] * fPrime(upperThetaJ[j]);
            }

            updateWeights();

            for (i = 0; i < outputNodes; i++)
            {
               actualOutputs[testCase][i] = F[i];
            }

            for (i = 0; i < outputNodes; i++)
            {
               totalError += 0.5 * lowerOmega[i] * lowerOmega[i];
            }
         } // for (testCase = 0; testCase < 4; testCase++)
         avgError = totalError / ((double) (numberOfCases));
      } // while (error > errorThreshold && iterations < maxIterations)
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
      }
   } // public void printTruthTable()

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

      if (shouldTrain)
      {
         System.out.println("\nOutputs");

         for (i = 0; i < outputNodes; i++)
         {
            System.out.print("F" + i + "     "); 
         }

         for (testCase = 0; testCase < numberOfCases; testCase++)
         {
            System.out.println();
            for (i = 0; i < outputNodes; i++)
            {
               System.out.printf("%.4f ", actualOutputs[testCase][i]);
            }
         }

         System.out.println("\n\nnumber of iterations: " + iterations);

         System.out.print("Reason for exiting: "); 
         
         if (iterations >= maxIterations)
         {
            System.out.println("Max iterations (" + maxIterations + ") reached");
         }
         else
         {
            System.out.printf("error under error threshold (%.4f)\n", errorThreshold);
         }
      } // if (train)

      System.out.printf("average error: %.4f\n", avgError);
   } // public void reportResults()

/**
 * Saves the calculated weights to a file for later usage
 * 
 * @throws IOException may be thrown when writing to the file fails 
 */
   public void saveWeights() throws IOException
   {
      int k, j, i;

      out = new PrintWriter(new FileWriter(filePath), true);
      
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

      in = new Scanner(new File(filePath));

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
 * The tester for the AB1 Network
 * 
 * @param args the parameters for the main method
 * @throws IOException 
 */
   public static void main(String args[]) throws IOException
   {
      ABCNetwork n = new ABCNetwork();
      
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
         n.run();
      }

      if (n.shouldSaveWeights)
      {
         n.saveWeights();
      }
      n.reportResults();
   } // public static void main(String args[])
} // public class AB1Network