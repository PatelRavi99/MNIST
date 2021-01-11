using System;
using System.Collections.Generic;
using System.Text;

namespace MNIST
{
    class Matrix
    {
        public double[,] matrix;
        public int[] size = new int[2];

        System.Random rand;

        public Matrix(double[,] m)
        {
            matrix = m;
            size[0] = m.GetLength(0);
            size[1] = m.GetLength(1);
        }

        public Matrix(int n, int m)
        {
            matrix = RandomMatrix(n, m);
            size[0] = n;
            size[1] = m;
        }

        public Matrix(int n)
        {
            matrix = new double[n, 1];

            for (int i = 0; i < n; i++)
            {
                matrix[i, 0] = 1;
            }

            size[0] = n;
            size[1] = 1;
        }

        private double[,] RandomMatrix(int n, int m)
        {
            double[,] mat = new double[n, m];
            rand = new System.Random();

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    mat[i, j] =NormalRand();
                }
            }

            return mat;
        }



        public Matrix Mult(Matrix matrix2)
        {
            double[,] newVals = new double[size[0], matrix2.size[1]];
            for (int i = 0; i < size[0]; i++)
            {
                for (int j = 0; j < matrix2.size[1]; j++)
                {
                    newVals[i, j] = DotProduct(GetRow(i), matrix2.GetCol(j));
                }
            }

            Matrix newMat = new Matrix(newVals);
            return newMat;
        }

        private double DotProduct(double[] v1, double[] v2)
        {
            double val = 0;
            for (int i = 0; i < v1.Length; i++)
            {
                val += v1[i] * v2[i];
            }

            return val;
        }

        private double[] GetRow(int r)
        {
            double[] row = new double[size[1]];

            for (int i = 0; i < size[1]; i++)
            {
                row[i] = matrix[r, i];
            }

            return row;
        }

        private double[] GetCol(int c)
        {
            double[] col = new double[size[0]];

            for (int i = 0; i < size[0]; i++)
            {
                col[i] = matrix[i, c];
            }

            return col;
        }

        public Matrix Add(Matrix matrix2)
        {
            double[,] newVals = new double[size[0], size[1]];
            for (int i = 0; i < size[0]; i++)
            {
                for (int j = 0; j < size[1]; j++)
                {
                    newVals[i, j] = matrix[i, j] + matrix2.matrix[i, j];
                }
            }

            Matrix newMat = new Matrix(newVals);
            return newMat;
        }

        public Matrix Sub(Matrix matrix2)
        {
            double[,] newVals = new double[size[0], size[1]];
            for (int i = 0; i < size[0]; i++)
            {
                for (int j = 0; j < size[1]; j++)
                {
                    newVals[i, j] = matrix[i, j] - matrix2.matrix[i, j];
                }
            }

            Matrix newMat = new Matrix(newVals);
            return newMat;
        }

        public Matrix MultScal(double s)
        {
            double[,] prod = new double[size[0], size[1]];
            for (int i = 0; i < size[0]; i++)
            {
                for (int j = 0; j < size[1]; j++)
                {
                    prod[i, j] = matrix[i, j] * s;
                }
            }

            Matrix product = new Matrix(prod);
            return product;
        }

        public Matrix MultDot(Matrix x)
        {
            double[,] vec = new double[x.size[0], x.size[1]];

            for (int i = 0; i < size[0]; i++)
            {
                for (int j = 0; j < size[1]; j++)
                {
                    vec[i, j] = matrix[i, j] * x.matrix[i, j];
                }
            }

            Matrix vector = new Matrix(vec);
            return vector;
        }

        public Matrix Transpose()
        {
            double[,] trans = new double[size[1], size[0]];

            for (int i = 0; i < size[1]; i++)
            {
                for (int j = 0; j < size[0]; j++)
                {
                    trans[i, j] = matrix[j, i];
                }
            }

            Matrix transpose = new Matrix(trans);
            return transpose;
        }

        public Matrix ZeroMatrix()
        {
            Matrix m = new Matrix(size[0], size[1]);

            for (int i = 0; i < size[0]; i++)
            {
                for (int j = 0; j < size[1]; j++)
                {
                    m.matrix[i, j] = 0;
                }
            }

            return m;
        }

        public double VectorDot(Matrix b)
        {
            double val = 0;
            for (int i = 0; i < size[0]; i++)
            {
                val += matrix[i, 0] * b.matrix[i, 0];
            }

            return val;
        }

        public void MatrixToString()
        {
            for (int i = 0; i < size[0]; i++)
            {
                for (int j = 0; j < size[1]; j++)
                {
                    Console.Write(matrix[i, j] + " ");
                }

                Console.WriteLine("\n");
            }
        }

        public Matrix copy()
        {
            double[,] m = new double[size[0], size[1]];
            for (int i = 0; i < m.GetLength(0); i++)
            {
                for (int j = 0; j < m.GetLength(1); j++)
                {
                    m[i, j] = matrix[i,j];
                }
            }

            Matrix newMat = new Matrix(m);
            return newMat;
        }

        public int LargestVecIndex()
        {
            int ind = 0;
            for (int i = 0; i < size[0]; i++)
            {
                if(matrix[i,0] > matrix[ind,0])
                {
                    ind = i;
                }
            }

            return ind;
        }

        public double SquaredSum()
        {
            double s = 0;
            for (int i = 0; i < size[0]; i++)
            {
                for (int j = 0; j < size[1]; j++)
                {
                    s += Math.Pow(matrix[i, j],2);
                }
            }

            return s;
        }

        public double NormalRand()
        {
            double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal = randStdNormal; //random normal(mean,stdDev^2)

            return randNormal;
        }

    }
}
