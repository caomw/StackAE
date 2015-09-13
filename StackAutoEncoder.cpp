/*******************************************************************
 *  Copyright(c) 2015
 *  All rights reserved.
 *
 *  Name: Stack Auto Encoder
 *  Description: A stacked autoencoder is a neural network consisting of multiple layers of sparse autoencoders in which the outputs of each layer is wired to the inputs of the successive layer.
 *
 *  Date: 2015-9-13
 *  Author: Yang
 *  Intruction: Use 50000 handwritten digit images to train the MLP and
 test with 10000 images.  Data from MNIST.
 
 ******************************************************************/

#include "StackAutoEncoder.h"

StackAutoEncoder::StackAutoEncoder(const int layers, const int rows, const int cols, const int labels, const int testsamples):
n_samples(rows), n_intput_dim(cols), n_layers(layers), n_output_labels(labels), n_test_samples(testsamples)
{
    /*
     *  Description:
     *  A constructed function to initialize the SAE
     *
     *  @param nLayers: total layers including input and output
     *  @param cols: input vector dimension(e.g. MNIST 784)
     *  @param rows: number of train samples
     *  @param labels: output vector dimension(e.g. MNIST 10)
     *  @param testsamples: number of test samples
     *
     */
    
    LoadTrainData();
    LoadTrainLabel();
    LoadTestLabel();
    LoadTestData();
    
    std::cout << "Please input hidden units respectively" << std::endl;
    for (int i = 0; i < n_layers; i++)
    {
        int units;
        std::cin >> units;
        m_hidden_units.push_back(units);
    }
    
    AutoEncoder* AE;
    AE = new AutoEncoder(m_train_data, m_hidden_units[0], DEFAULT_SPARSITY);
    m_AELayers.push_back(AE);
    for (int i = 1; i < layers; i++)
    {
        AE = new AutoEncoder(m_AELayers[i - 1]->GetHiddenMatrix(),m_hidden_units[i], DEFAULT_SPARSITY);
        m_AELayers.push_back(AE);
    }
}

StackAutoEncoder::~StackAutoEncoder()
{
    m_hidden_units.clear();
    m_AELayers.clear();
}

void StackAutoEncoder::LoadTrainData()
{
    /*
     *  Description:
     *  Load train data from MNIST, normalize original pixel from 256
     *  to 1
     *
     */
    
    std::string str = "/Users/liuyang/Desktop/Class/ML/DL/ANN/MNIST/train_x";
    
    LoadMatrix(m_train_data, str, n_intput_dim, n_samples);
    m_train_data /= 256.;
}

void StackAutoEncoder::LoadTrainLabel()
{
    /*
     *  Description:
     *  Load train label from MNIST (1-of-k scheme)
     *
     */
    
    std::string str = "/Users/liuyang/Desktop/Class/ML/DL/ANN/MNIST/train_y";
    
    LoadMatrix(m_train_label, str, n_output_labels, n_samples);
}

void StackAutoEncoder::LoadTestData()
{
    /*
     *  Description:
     *  Load test data from MNIST, normalize original pixel from 256
     *  to 1
     *
     */
    std::string str = "/Users/liuyang/Desktop/Class/ML/DL/ANN/MNIST/test_x";
    
    LoadMatrix(m_test_data, str, n_intput_dim, n_test_samples);

    m_test_data /= 256.;
}

void StackAutoEncoder::LoadTestLabel()
{
    /*
     *  Description:
     *  Load test label from MNIST (1-of-k scheme)
     *
     */
    std::string str = "/Users/liuyang/Desktop/Class/ML/DL/ANN/MNIST/test_y";
    
    LoadMatrix(m_test_label, str, n_output_labels, n_test_samples);
}

void StackAutoEncoder::FeedForward()
{
    /*
     *  Description:
     *  Feed forward in the auto encoder to get feature
     *
     */
    for (int i = 0; i < n_layers; i++)
    {
        for (int j = 0; j < n_samples; j++)
        {
            m_AELayers[i]->Calculate(j);
            m_AELayers[i]->BackPropagate(j, 0.7);
        }
    }
    auto lit = m_AELayers.end() - 1;
    m_train_feature = (*lit)->GetHiddenMatrix();
}

void StackAutoEncoder::Test()
{
    /*
     *  Description:
     *  Fine tuning using NN
     *
     */
    
    /* Load the weight matrix and bias into NN */
    std::vector<MatrixXf> Matrixs;
    std::vector<VectorXf> Biases;
    for (int i = 0; i < n_layers; i++)
    {
        Matrixs.push_back(m_AELayers[i]->GetWeight_encode());
        Biases.push_back(m_AELayers[i]->GetBias_encode());
    }
    
    std::vector<unsigned int> Units;
    Units = m_hidden_units;
    
    Units.insert(Units.begin(), n_intput_dim);
    Units.push_back(n_output_labels);
    
    NeuralNetwork NN(n_layers + 2, Units);
    NN.LoadWeight(Matrixs, Biases);
    
    for (int i = 0; i < n_samples; i++)
    {
        VectorXf iFeature = (m_train_data.row(i)).transpose();
        VectorXf oLabel = (m_train_label.row(i)).transpose();
        NN.Backpropagate(oLabel, NN.Calculate(iFeature));
    }
    
    /* Start test */
    double err = 0;
    double invalid = 0;
    for (int i = 0; i < n_test_samples; i++)
    {
        VectorXf iFeature = (m_test_data.row(i)).transpose();
        VectorXf oLabel = (m_test_label.row(i)).transpose();
        double err_x = (NN.Calculate(iFeature) - oLabel).sum();
        
        if (err_x > 1)
        {
            err++;
        }
        else if (err_x < 1 && err_x > 0.1)
        {
            invalid++;
        }
    }
    
    std::cout << "The error numbers of 10000 test sets is " << err << std::endl;
    std::cout << "The abandom numbers of 10000 test sets is " << invalid << std::endl;
}