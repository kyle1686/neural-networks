import java.util.Scanner;

/**
 * The Parser class uses a Scanner to parse configuration files used to run the ABC Backprop network
 * 
 * @author  Kyle Li
 * @verison 27 March 2024
 * Date of creation: 24 March 2024
 */
public class NNParser
{
   Scanner sc;

   int inputNodes;
   int hiddenLayerNodes;
   int outputNodes;
   int maxIterations;
   int numberOfCases; 

   double errorThreshold; 
   double lambda; 
   double low;
   double high;

   boolean shouldTrain;
   boolean shouldSaveWeights;
   boolean useRandomWeights;

   String weightsFilePath;

   double testCases[][]; 
   double trueOutputs[][];
   
/**
 * Constructor for the NNParser class that initializes the scanner
 * 
 * @param s the scanner to be passed in
*/
   public NNParser(Scanner s)
   {
      sc = s;
   }

/**
 * Parses the configuration file by line, sifting through each string to get the variable name and the value it should be assigned to
*/
   public void parseConfigFile()
   {
      int testCase, k, i, it; 
      String line; 

      sc.nextLine();
      line = sc.nextLine();

      while (!line.equals("TRUTH TABLE"))
      {
         String varName = "";

         it = 0;
         String c = line.substring(it, it + 1);

         while (!c.equals(" "))
         {
            it++; 
            varName += c;
            c = line.substring(it, it + 1);
         }

         it += 2;
         
         String val = ""; 

         while (it < line.length() - 1)
         {
            it++;
            c = line.substring(it, it + 1);
            val += c;
         }

         try
         {
            double numVal = Double.parseDouble(val);

            if (varName.equals("inputNodes"))
               inputNodes = (int) numVal;    

            else if (varName.equals("hiddenLayerNodes"))
               hiddenLayerNodes = (int) numVal;
               
            else if (varName.equals("outputNodes"))
               outputNodes = (int) numVal;

            else if (varName.equals("maxIterations"))
               maxIterations = (int) numVal;

            else if (varName.equals("numberOfCases"))
               numberOfCases = (int) numVal;

            else if (varName.equals("lambda"))
               lambda = numVal; 

            else if (varName.equals("errorThreshold"))
               errorThreshold = numVal; 

            else if (varName.equals("low"))
               low = numVal; 

            else if (varName.equals("high"))
               high = numVal; 
         } // try
         catch (NumberFormatException e)
         {
            int boolVal = -1;

            if (val.equals("true"))
               boolVal = 1;

            else if (val.equals("false"))
               boolVal = 0;


            if (boolVal != -1)
            {
               if (varName.equals("shouldTrain"))
                  shouldTrain = (boolVal == 1); 
               
               else if (varName.equals("shouldSaveWeights"))
                  shouldSaveWeights = (boolVal == 1); 

               else if (varName.equals("useRandomWeights"))
                  useRandomWeights = (boolVal == 1); 
            } // if (boolVal != -1)
            else
            {
               weightsFilePath = val;
            }
         } // catch (NumberFormatException e)
         line = sc.nextLine();
      } // while (!line.equals("TRUTH TABLE"))

      testCases = new double[numberOfCases][inputNodes];
      trueOutputs = new double[numberOfCases][outputNodes]; 

      for (testCase = 0; testCase < numberOfCases; testCase++)
      {
         for (k = 0; k < inputNodes; k++)
         {
            testCases[testCase][k] = sc.nextDouble(); 
         }

         for (i = 0; i < outputNodes; i++)
         {
            trueOutputs[testCase][i] = sc.nextDouble(); 
         }

         if (sc.hasNextLine())
            sc.nextLine();
      } // for (testCase = 0; testCase < numberOfCases; testCase++)
   } // public void parseConfigFile()

/**
 * Prints out the values in all the variables (for testing / debugging), as well as the truth table 
 */
   public void printEverything()
   {
      int k, i, testCase;

      System.out.println("inputNodes = " + inputNodes);
      System.out.println("hiddenLayerNodes = " + hiddenLayerNodes);
      System.out.println("outputNodes = " + outputNodes);
      System.out.println("maxIterations = " + maxIterations);
      System.out.println("numberOfCases = " + numberOfCases);
      System.out.println("lambda = " + lambda);
      System.out.println("errorThreshold = " + errorThreshold);
      System.out.println("low = " + low);
      System.out.println("high = " + high);
      System.out.println("shouldTrain = " + shouldTrain);
      System.out.println("shouldSaveWeights = " + shouldSaveWeights);
      System.out.println("useRandomWeights = " + useRandomWeights);
      System.out.println("weightsFilePath = " + weightsFilePath);

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
   } // public void printEverything()
} // public class NNParser