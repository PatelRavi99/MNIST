using System;
using System.Collections.Generic;
using System.Text;

namespace MNIST
{
    class NeuralNetwork
    {
        public int[,] weightShapes;
        public Matrix[] weights;
        public Matrix[] biasis;

        public Matrix[] a;
        public Matrix[] z;

        public Matrix[] error;

        System.Random rand = new System.Random();

        public int numLayers;

        public NeuralNetwork(Matrix layers)
        {
            numLayers = layers.matrix.GetLength(1);
            weightShapes = new int[numLayers - 1, 2];
            weights = new Matrix[numLayers - 1];
            biasis = new Matrix[numLayers - 1];

            a = new Matrix[numLayers - 1];
            z = new Matrix[numLayers - 1];

            //Constructing weight shapes
            for (int i = 1; i < numLayers; i++)
            {
                weightShapes[i - 1, 0] = (int)layers.matrix[0, i];
                weightShapes[i - 1, 1] = (int)layers.matrix[0, i - 1];
            }

            //Constructing random weights
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                Matrix m = new Matrix(weightShapes[i, 0], weightShapes[i, 1]);
                weights[i] = m;
            }

            //Constructing biasis
            for (int i = 1; i < numLayers; i++)
            {
                Matrix b = new Matrix((int)layers.matrix[0, i]);
                biasis[i - 1] = b;
            }
        }

        private Matrix ActivationFunction(Matrix x)
        {
            Matrix theta = x.copy();

            for (int i = 0; i < theta.size[0]; i++)
            {
                for (int j = 0; j < theta.size[1]; j++)
                {
                    theta.matrix[i, j] = 1 / (1 + Math.Exp(-(float)theta.matrix[i, j]));
                }
            }

            return theta;
        }

        private Matrix DActivationFunction(Matrix x)
        {
            Matrix theta = ActivationFunction(x.copy());

            for (int i = 0; i < theta.size[0]; i++)
            {
                for (int j = 0; j < theta.size[1]; j++)
                {
                    theta.matrix[i, j] = theta.matrix[i, j] * (1 - theta.matrix[i, j]);
                }
            }

            return theta;
        }

        public Matrix predict(Matrix input)
        {
            z[0] = weights[0].Mult(input).Add(biasis[0]);
            a[0] = ActivationFunction(z[0]);
            for (int i = 1; i < weightShapes.GetLength(0); i++)
            {
                
                z[i] = weights[i].Mult(a[i-1]).Add(biasis[i]);
                a[i] = ActivationFunction(z[i]);
            }

            return a[weightShapes.GetLength(0) - 1];
        }

        private void calculateError(Matrix y)
        {
            error = new Matrix[numLayers - 1];

            error[numLayers - 2] = y.Sub(a[weightShapes.GetLength(0) - 1]);
            for (int i = numLayers - 3; i >= 0; i--)
            {
                error[i] = weights[i + 1].Transpose().Mult(error[i + 1]);
            }
        }

        private Matrix[] GradientCalc(Matrix inp)
        {
            Matrix[] grad = new Matrix[numLayers - 1];

            grad[0] = error[0].MultDot(DActivationFunction(z[0]));
            biasis[0] = biasis[0].Add(grad[0].MultScal(0.001));
            grad[0] = grad[0].Mult(inp.copy().Transpose());
            for (int i = 1; i < error.Length; i++)
            {
                grad[i] = error[i].MultDot(DActivationFunction(z[i]));
                biasis[i] = biasis[i].Add(grad[i].MultScal(0.001));
                grad[i] = grad[i].Mult(a[i - 1].Transpose());
            }

            return grad;
        }

        private Matrix[] initGrad()
        {
            Matrix[] grads = new Matrix[weights.Length];

            for (int i = 0; i < weights.Length; i++)
            {
                Matrix temp = weights[i];
                grads[i] = temp.ZeroMatrix();
            }

            return grads;
        }

        public void TrainNetwork(Matrix[] TrainingImages, Matrix[] TrainingLabels)
        {
            double alpha = 0.6;
            double eta = 0.3;
            double decay = 0.01;

            int epochs = 75;
            Matrix[] prevGrad = initGrad();
            Matrix[] deltaWieghts = new Matrix[weights.Length];
            Matrix[] gradients;

            double maxError = 0;

            for (int i = 0; i < epochs; i++)
            {
                //alpha /= (1 + i * alpha);
                for (int j = 0; j < 60000; j++)
                {
                    Matrix output;
                    output = predict(TrainingImages[j].copy()).copy();

                    if(output.Sub(TrainingLabels[j]).SquaredSum() > maxError)
                    {
                        maxError = output.Sub(TrainingLabels[j]).SquaredSum();
                    }

                    calculateError(TrainingLabels[j].copy());
                    gradients = GradientCalc(TrainingImages[j].copy());
                    for (int k = 0; k < deltaWieghts.Length; k++)
                    {
                        deltaWieghts[k] = gradients[k].MultScal(eta).Add(prevGrad[k].MultScal(alpha));
                        weights[k] = weights[k].Add(deltaWieghts[k]);
                        prevGrad[k] = gradients[k].copy();

                    }

                    if(j % 10000 == 0)
                    {
                        Console.WriteLine("Max Error: " + maxError);
                        maxError = 0;
                    }

                    //Console.WriteLine("image #" + j + " epoch #" + i);
                }

                Console.WriteLine("epoch #" + i + " complete!");

                Matrix[][] shuf = shuffle(TrainingImages, TrainingLabels);
                TrainingImages = shuf[0];
                TrainingLabels = shuf[1];
            }

        }

        public double TestNetwork(Matrix[] TestingImages, Matrix[] TestingLabels)
        {
            double counter = 0;
            for (int i = 0; i < TestingImages.Length; i++)
            {
                Matrix output = predict(TestingImages[i].copy()).copy();

                if(output.LargestVecIndex() == TestingLabels[i].LargestVecIndex())
                {
                    counter += 1;
                }
            }

            Console.WriteLine("Correct Predictions: " + counter);
            double acc = counter / (double)TestingImages.Length;
            return (acc);
        }

        private Matrix[][] shuffle(Matrix[] trainingIamges, Matrix[] trainingLables)
        {
            for (int i = 0; i < 100000; i++)
            {
                int temp1 = rand.Next(0, 60000);
                int temp2 = rand.Next(0, 60000);

                Matrix t = trainingIamges[temp2];
                trainingIamges[temp2] = trainingIamges[temp1];
                trainingIamges[temp1] = t;

                Matrix p = trainingLables[temp2];
                trainingLables[temp2] = trainingLables[temp1];
                trainingLables[temp1] = p;
            }

            Matrix[][] shuf = new Matrix[2][];
            shuf[0] = trainingIamges;
            shuf[1] = trainingLables;

            return shuf;
        }
    }
}
