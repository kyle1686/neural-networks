/**
 * This abstract class describes any function, with methods describing the activation function and its derivative
 * 
 * @author Kyle Li
 * @version 22 April 2024
 * Date of creation: April 20
 */
public abstract class ActivationFunction 
{
/**
 * The activation function used for the network. Helps caluclate the hidden layer and output layer activations.
 * 
 * @param x activations in previous layer * weights, or uppercase theta
 * @return  the value of the activation
 */
   public abstract double f(double x); 
/**
 * The derivative of the activation function, i.e the change in the activation function. Returns the slope of f at the given
 * x value. 
 * 
 * @param x  activations in previous layer * weights, or uppercase theta
 * @return  the change in the activation function
 */
   public abstract double fPrime(double x);
} // public abstract class ActivationFunction 
