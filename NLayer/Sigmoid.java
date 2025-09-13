/**
 * A sigmoid is a possible activation function used for the network. This class describes the sigmoid function
 * and its derivative. 
 * 
 * Table of Contents
 *  • public double f(double x)
 *  • public double fPrime(double x)
 * 
 * @author Kyle Li
 * @version 30 April 2024
 * Date of creation: April 20
 */
public class Sigmoid extends ActivationFunction
{
/**
 * The activation function (sigmoid) used for the network. Helps calculate the hidden layer and output layer activations.
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
} // public class Sigmoid extends ActivationFunction
