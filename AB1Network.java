/**
 * The A-B-1 Network is a multi-layer perceptron that uses gradient descent to minimize the error across all test cases. 
 * With every iteration, it takes a 'step downhill' the error curve and optimizes its existing weights to reduce the error 
 * as much as possible for every iteration. The network is based off a feed-forward model. 
 * 
 * @author  Kyle Li
 * @version 24 February 2024
 * Date of creation: 1 February 2024
 */
public class AB1Network
{
   final int INPUT_NODES = 2;
   final int HIDDEN_LAYER_NODES = 1; 
   final int OUTPUT_NODES = 1;

   final double ERROR_THRESHOLD = 0.0002;
   
   final int MAX_ITERATIONS = 100000;

   final double LEARNING_RATE = 0.3;

   final int NUMBER_OF_CASES = 4;

   final double LOWER_BOUND = -1.5;
   final double UPPER_BOUND = 1.5;

   int inputNodes;
   int hiddenLayerNodes;
   int outputNodes;

   double a[];      // the array of input activations
   double h[];      // the array of hidden activations
   double F0;       // the output activation
   
   double upperTheta[];
   double upperOmega[];
   double lowerOmega[];
   double upperPsi[];
   double lowerPsi[];

   double totalError;
   double avgError; 
   double errorThreshold; 

   double weightsKJ[][];
   double weightsJ0[][];

   double lambda; 

   double changeInWeightsKJ[][];
   double changeInWeightsJ0[][];

   double changeInErrorKJ[][];
   double changeInErrorJ0[][];

   int iterations;
   int maxIterations;

   int numberOfCases; 
   double testCases[][]; 
   double trueOutputs[];
   double actualOutputs[];

   double low;
   double high;

   boolean useRandomWeights;
   boolean shouldTrain;

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

      shouldTrain = false;
      useRandomWeights = true;
   } // public void setConfigParams()

/**
 * Prints all configuration paramaters to the terminal. 
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

      weightsKJ = new double[inputNodes][hiddenLayerNodes];
      weightsJ0 = new double[hiddenLayerNodes][1];

      testCases = new double[numberOfCases][inputNodes];
      trueOutputs = new double[numberOfCases]; 
      actualOutputs = new double[numberOfCases];
      
      if (shouldTrain)
      {  
         upperTheta = new double[hiddenLayerNodes];

         upperOmega = new double[hiddenLayerNodes];
         lowerOmega = new double[hiddenLayerNodes];
         upperPsi = new double[hiddenLayerNodes];
         lowerPsi = new double[hiddenLayerNodes];

         changeInWeightsKJ = new double[inputNodes][hiddenLayerNodes];
         changeInWeightsJ0 = new double[hiddenLayerNodes][1];

         changeInErrorKJ = new double[inputNodes][hiddenLayerNodes];
         changeInErrorJ0 = new double[hiddenLayerNodes][1];
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
 * Creates all test cases and their respective answers manually and stores them in arrays for future use 
 */
   public void createTruthTable()
   {
      testCases[0][0] = 0.0;
      testCases[0][1] = 0.0;
      trueOutputs[0] = 0.0;

      testCases[1][0] = 0.0;
      testCases[1][1] = 1.0;
      trueOutputs[1] = 1.0;

      testCases[2][0] = 1.0;
      testCases[2][1] = 0.0;
      trueOutputs[2] = 1.0;

      testCases[3][0] = 1.0;
      testCases[3][1] = 1.0;
      trueOutputs[3] = 0.0;
   } // public void createTruthTable()

/**
 * Randomizes the weights for the KJ and J0 layers in a range between low (the lower bound of the range) and 
 * high (the upper bound of the range)
*/
   public void randomizeWeights()
   {
      int j, k;

      for (k = 0; k < inputNodes; k++)
      {
         for (j = 0; j < hiddenLayerNodes; j++)
         {
            weightsKJ[k][j] = randomize();
         }
      }

      for (j = 0; j < hiddenLayerNodes; j++)
      {
         weightsJ0[j][0] = randomize();
      }
   } // public void randomizeWeights()

   /**
    * Manually sets the weights instead of randomizing
    */
   public void setManualWeights()
   {
      int j, k;

      for (k = 0; k < inputNodes; k++)
      {
         for (j = 0; j < hiddenLayerNodes; j++)
         {
            weightsKJ[k][j] = 0;
         }
      }

      for (j = 0; j < hiddenLayerNodes; j++)
      {
         weightsJ0[j][0] = 0;
      }
   } // Manually sets the weights instead of randomizing

/**
 * Sets the test cases and the expected outputs for each case, and randomizes the weights in the KJ and 
 * J0 layers
 */
   public void populateArrays()
   {
      createTruthTable();

      if (useRandomWeights)
      {
         randomizeWeights();
      }
      else
      {
         setManualWeights();
      }
   } // public void populateArrays()

/**
 * Calculates the change in error, the change in weights, and updates the actual weights in both the KJ 
 * layer and the J0 layer between a range. 
 */
   public void updateWeights()
   {
      int j, k;

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
         changeInErrorJ0[j][0] = -h[j] * lowerPsi[0];
         changeInWeightsJ0[j][0] = -lambda * changeInErrorJ0[j][0];
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
         weightsJ0[j][0] += changeInWeightsJ0[j][0]; 
      }
   } // public void updateWeights()

/**
 * Calculates the value of the hidden activation(s) and the final output activation when training
 */
   public void calculateActivationsForTrain()
   {
      for (int j = 0; j < hiddenLayerNodes; j++)
      {
         upperTheta[j] = 0.0;
      }

      for (int k = 0; k < inputNodes; k++)
      {
         for (int j = 0; j < hiddenLayerNodes; j++)
         {
            upperTheta[j] += a[k] * weightsKJ[k][j];
         }
      }

      for (int j = 0; j < hiddenLayerNodes; j++)
      {
         h[j] = f(upperTheta[j]);
      }

      upperTheta[0] = 0.0;
      for (int j = 0; j < hiddenLayerNodes; j++)
      {
         upperTheta[0] += h[j] * weightsJ0[j][0];
      }

      F0 = f(upperTheta[0]);
   } // public void calculateActivationsForTrain()

/**
 * Calculates the value of the hidden activation(s) and the final output activation when running
 */
   public void calculateActivationsForRun()
   {
      for (int k = 0; k < inputNodes; k++)
      {
         for (int j = 0; j < hiddenLayerNodes; j++)
         {
            h[j] += f(a[k] * weightsKJ[k][j]);
         }
      }

      h[0] = 0.0;
      for (int j = 0; j < hiddenLayerNodes; j++)
      {
         h[0] += h[j] * weightsJ0[j][0];
      }

      F0 = f(h[0]);
   } // public void calculateActivationsForRun()

/**
 * Runs the network as it is on all test cases; unlike train, run does not optimize the weights and immediately
 * calculates the output activation
 */
   public void run()
   {
      int testCase, inputs;

      for (testCase = 0; testCase < numberOfCases; testCase++)
      {
         for (inputs = 0; inputs < inputNodes; inputs++)
         {
            a[inputs] = testCases[testCase][inputs];
         }
            
         calculateActivationsForRun();

         actualOutputs[testCase] = F0;

         totalError += 0.5 * (trueOutputs[testCase] - F0) * (trueOutputs[testCase] - F0);
      } // for (testCase = 0; testCase < numberOfCases; testCase++)
      avgError = totalError / (double) numberOfCases;
   } // public void run()

/**
 * Repeatedly iterates through, modifying the weights on each pass to better match the output activation to the 
 * expected output. Finishes training when the current error is under the error threshold, or when the maximum
 * number of iterations has been reached
 */
   public void train()
   {
      int j, testCase, inputs; 

      totalError = Double.MAX_VALUE;
      avgError = totalError / (double) numberOfCases;

      while (avgError > errorThreshold && iterations < maxIterations)
      {
         totalError = 0.0;

         iterations++;

         for (testCase = 0; testCase < numberOfCases; testCase++)
         {
            for (inputs = 0; inputs < inputNodes; inputs++)
            {
               a[inputs] = testCases[testCase][inputs];
            }

            calculateActivationsForTrain();

            lowerOmega[0] = (trueOutputs[testCase]) - F0; 
            lowerPsi[0] = lowerOmega[0] * fPrime(upperTheta[0]);

            for (j = 0; j < hiddenLayerNodes; j++)
            {
               upperOmega[j] = lowerPsi[0] * weightsJ0[j][0];
               upperPsi[j] = upperOmega[j] * fPrime(upperTheta[j]);
            }

            updateWeights();

            actualOutputs[testCase] = F0;

            totalError += 0.5 * ((trueOutputs[testCase]) - F0) * ((trueOutputs[testCase]) - F0);
         } // for (testCase = 0; testCase < 4; testCase++)
         avgError = totalError / (double) numberOfCases;
      } // while (error > errorThreshold && iterations < maxIterations)
   } // public void train()

/**
 * Prints the truth table, including all cases and their expected values. 
 */
   public void printTruthTable()
   {
      int testCase, inputs;

      System.out.println("Truth table");
      System.out.println("——————————————");

      for (inputs = 0; inputs < inputNodes; inputs++)
      {
         System.out.print("a" + (inputs + 1) + "  ");
      }

      System.out.println("   F0");
      System.out.println("——————————————");

      for (testCase = 0; testCase < numberOfCases; testCase++)
      {
         for (inputs = 0; inputs < inputNodes; inputs++)
         {
            System.out.print(testCases[testCase][inputs] + " ");
         }
         System.out.println(" | " + trueOutputs[testCase]);
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
      int testCase;

      printTruthTable();

      if (shouldTrain)
      {
         for (testCase = 0; testCase < numberOfCases; testCase++)
         {
            System.out.printf("case " + (testCase + 1) + " actual value: ");
            System.out.printf("%.8f\n", actualOutputs[testCase]);
         }                                 

         System.out.println("number of iterations: " + iterations);

         System.out.print("Reason for exiting: "); 
         if (iterations >= maxIterations)
         {
            System.out.println("Max iterations (" + maxIterations + ") reached");
         }
         else
         {
            System.out.printf("error under error threshold (%.4f", errorThreshold);
            System.out.println(")");
         }
      } // if (train)

      System.out.printf("average error: %.4f\n", avgError);
   } // public void reportResults()

/**
 * The tester for the AB1 Network
 * 
 * @param args the parameters for the main method
 */
   public static void main(String args[])
   {
      AB1Network n = new AB1Network();
      
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
      n.reportResults();
   } // public static void main(String args[])
} // public class AB1Network