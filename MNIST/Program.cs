using System;
using System.IO;

namespace MNIST
{
    class Program
    {
        public static Matrix[] trainingIamges;
        public static Matrix[] trainingLables;

        public static Matrix[] testingImages;
        public static Matrix[] testingLabels;

        static void Main(string[] args)
        {
            LoadData();

            //trainingLables[0].MatrixToString();

            double[,] l = new double[1, 4] { { 784, 75, 75 , 10 } };
            Matrix layers = new Matrix(l);

            NeuralNetwork nn = new NeuralNetwork(layers);

            Console.WriteLine("Training Network!");
            nn.TrainNetwork(trainingIamges, trainingLables);
            Console.WriteLine("Training Network Complete!");

            Console.WriteLine("Testing Network!");
            Console.WriteLine("Accuracy of Network: " + (double)nn.TestNetwork(trainingIamges, trainingLables));
            Console.WriteLine("Testing Network Complete!");

            SaveWeights(nn.weights,nn.biasis);
            
        }

        public static void LoadData()
        {
            Console.WriteLine("Loading Data!");

            LoadTrainingData(@"C:\Users\ravpa\OneDrive\Desktop\train-labels.idx1-ubyte", @"C:\Users\ravpa\OneDrive\Desktop\train-images.idx3-ubyte");
            ShuffleTrainingData();

            LoadTestingData(@"C:\Users\ravpa\OneDrive\Desktop\t10k-labels.idx1-ubyte", @"C:\Users\ravpa\OneDrive\Desktop\t10k-images.idx3-ubyte");

            Console.WriteLine("Done Loading!");
        }

        private static void LoadTrainingData(string pathLabels, string pathImages)
        {
            FileStream images = new FileStream(pathImages, FileMode.Open);
            FileStream labels = new FileStream(pathLabels, FileMode.Open);

            BinaryReader binImages = new BinaryReader(images);
            BinaryReader binLabels = new BinaryReader(labels);

            byte[] magicNum1 = binImages.ReadBytes(4);
            byte[] numImages = binImages.ReadBytes(4);
            byte[] numRows = binImages.ReadBytes(4);
            byte[] numCols = binImages.ReadBytes(4);

            byte[] magicNum2 = binLabels.ReadBytes(4);
            byte[] numLabels = binLabels.ReadBytes(4);

            trainingIamges = new Matrix[10000];
            trainingLables = new Matrix[10000];

            for (int i = 0; i < 60000; i++)
            {
                double[,] lab = new double[10, 1] { { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 } };
                lab[(int)binLabels.ReadByte(), 0] = 1;

                Matrix Lable = new Matrix(lab);
                trainingLables[i] = Lable;

                double[,] img = new double[784, 1];
                for (int j = 0; j < 28; j++)
                {
                    for (int k = 0; k < 28; k++)
                    {
                        img[(j * 28) + k, 0] = (int)binImages.ReadByte();
                    }
                }

                Matrix imgBytes = new Matrix(img);
                trainingIamges[i] = imgBytes;
            }

        }

        private static void LoadTestingData(string pathLabels, string pathImages)
        {
            FileStream images = new FileStream(pathImages, FileMode.Open);
            FileStream labels = new FileStream(pathLabels, FileMode.Open);

            BinaryReader binImages = new BinaryReader(images);
            BinaryReader binLabels = new BinaryReader(labels);

            byte[] magicNum1 = binImages.ReadBytes(4);
            byte[] numImages = binImages.ReadBytes(4);
            byte[] numRows = binImages.ReadBytes(4);
            byte[] numCols = binImages.ReadBytes(4);

            byte[] magicNum2 = binLabels.ReadBytes(4);
            byte[] numLabels = binLabels.ReadBytes(4);

            testingImages = new Matrix[10000];
            testingLabels = new Matrix[10000];

            for (int i = 0; i < 10000; i++)
            {
                double[,] lab = new double[10, 1] { { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 } };
                lab[(int)binLabels.ReadByte(), 0] = 1;

                Matrix Lable = new Matrix(lab);
                testingLabels[i] = Lable;

                double[,] img = new double[784, 1];
                for (int j = 0; j < 28; j++)
                {
                    for (int k = 0; k < 28; k++)
                    {
                        img[(j * 28) + k, 0] = (int)binImages.ReadByte();
                    }
                }

                Matrix imgBytes = new Matrix(img);
                testingImages[i] = imgBytes;
            }
        }
        private static void ShuffleTrainingData()
        {
            System.Random rand = new System.Random();

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
        }

        private static void SaveWeights(Matrix[] weights, Matrix[] biasis)
        {
            string info = "";

            for (int i = 0; i < weights.Length; i++)
            {
                info += weights[i].size[0].ToString() + " " + (string)weights[i].size[1].ToString() + "\n";

                for (int j = 0; j < weights[i].size[0]; j++)
                {
                    for (int k = 0; k < weights[i].size[1]; k++)
                    {
                        info += weights[i].matrix[j, k].ToString() + " ";
                    }
                    info += "\n";
                }
            }

            for (int i = 0; i < biasis.Length; i++)
            {
                info += biasis[i].size[0].ToString() + " " + (string)biasis[i].size[1].ToString() + "\n";

                for (int j = 0; j < biasis[i].size[0]; j++)
                {
                    for (int k = 0; k < biasis[i].size[1]; k++)
                    {
                        info += biasis[i].matrix[j, k].ToString() + " ";
                    }
                    info += "\n";
                }
            }

            string path = @"C:\Users\ravpa\OneDrive\Desktop\weights.txt";
            File.WriteAllText(path, info);
        }
    }
}
