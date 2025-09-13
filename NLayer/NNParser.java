import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

/**
 * The Parser class uses a Scanner to parse configuration files used to run the N-Layer network
 * 
 * Table of Contents
 *  • public void parseConfigFile()
 *  • public void printEverything()
 * 
 * @author  Kyle Li
 * @verison 30 April 2024
 * Date of creation: 24 March 2024
 */
public class NNParser
{
   static final int INPUTLAYER = 0;

   Scanner sc;

   int n;

   int layers[]; 
   int maxIterations;
   int numberOfCases; 

   double errorThreshold; 
   double lambda; 
   double low;
   double high;

   boolean shouldTrain;
   boolean shouldSaveWeights;
   boolean useRandomWeights;

   int keepAlive;

   String weightsFilePath;
   String truthTableFilePath;

   double testCases[][]; 
   double trueOutputs[][];

   ActivationFunction act;
   
/**
 * Constructor for the NNParser class that initializes the scanner
 * 
 * @param s the scanner to be passed in
*/
   public NNParser(Scanner s)
   {
      sc = s;
      n = 0;
   }

/**
 * Parses the configuration file by line, sifting through each string to get the variable name and the value it should be assigned to
 * @throws FileNotFoundException 
*/
   public void parseConfigFile() throws FileNotFoundException
   {
      int testCase, k, i, it, hiddenLayerNumber; 
      String line; 

      while (sc.hasNextLine())
      {
         line = sc.nextLine();

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

            if (varName.equals("n"))
            {
               n = (int) numVal; 
               layers = new int[n];
            }

            else if (varName.equals("inputNodes"))
               layers[INPUTLAYER] = (int) numVal;
               
            else if (varName.equals("outputNodes"))
               layers[n - 1] = (int) numVal;

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
            
            else if (varName.equals("keepAlive"))
               keepAlive = (int) numVal;

            else if (varName.substring(0, 16).equals("hiddenLayerNodes"))
            {
               hiddenLayerNumber = Integer.parseInt(varName.substring(16));
               layers[hiddenLayerNumber] = (int) numVal; 
            }
            else 
               System.out.println(varName.substring(0, 15));
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
               if (varName.equals("weightsFilePath"))
                  weightsFilePath = val;

               else if (varName.equals("activationFunction"))
               {
                  if (val.equals("SIGMOID"))
                     act = new Sigmoid();
               }

               else if (varName.equals("truthTableFilePath"))
                  truthTableFilePath = val;
            } // if (boolVal != -1)...else
         } // try...catch (NumberFormatException e)
      } // while (sc.hasNextLine())

      sc = new Scanner(new File(truthTableFilePath));

      testCases = new double[numberOfCases][layers[INPUTLAYER]];
      trueOutputs = new double[numberOfCases][layers[n - 1]]; 

      for (testCase = 0; testCase < numberOfCases; testCase++)
      {
         for (k = 0; k < layers[INPUTLAYER]; k++)
         {
            testCases[testCase][k] = sc.nextDouble(); 
         }

         for (i = 0; i < layers[n - 1]; i++)
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
      int k, i, testCase, curr;

      System.out.println("n = " + n);
      for (curr = 0; curr < n - 1; curr++)
      {
         System.out.print(layers[curr] + "-"); 
      }
      System.out.println(layers[n - 1] + " Network");
      System.out.println("maxIterations = " + maxIterations);
      System.out.println("numberOfCases = " + numberOfCases);
      System.out.println("lambda = " + lambda);
      System.out.println("errorThreshold = " + errorThreshold);
      System.out.println("low = " + low);
      System.out.println("high = " + high);
      System.out.println("shouldTrain = " + shouldTrain);
      System.out.println("shouldSaveWeights = " + shouldSaveWeights);
      System.out.println("useRandomWeights = " + useRandomWeights);
      System.out.println("keepAlive = " + keepAlive);
      System.out.println("weightsFilePath = " + weightsFilePath);
      System.out.println("truthTableFilePath = " + truthTableFilePath);

      for (testCase = 0; testCase < numberOfCases; testCase++)
      {
         for (k = 0; k < layers[INPUTLAYER]; k++)
         {
            System.out.print(testCases[testCase][k] + " ");
         }

         System.out.print("| ");

         for (i = 0; i < layers[n - 1]; i++)
         {
            System.out.print(trueOutputs[testCase][i] + " ");
         }
         System.out.println();
      } // for (testCase = 0; testCase < numberOfCases; testCase++)
   } // public void printEverything()
} // public class NNParser