#ifndef __ANN__ANN__
#define __ANN__ANN__

#include <vector>
#include <cmath>
#include "LoadMatrix.h"

#define TRAINSAMPLE 50000
#define TESTSAMPLE 10000
#define X_COLS 784
#define Y_COLS 10
#define DEFAULT_LAYERS 3
#define DEFAULT_LEARNING_RATE 0.7
#define DSIGMOID(x) x * (1 - x)

using Eigen::MatrixXf;
using Eigen::VectorXf;

class NNLayer;
typedef std::vector<NNLayer*> VectorLayers;

class NeuralNetwork
{
    typedef unsigned int UINT;
    typedef std::vector<UINT> n_Neurons;
    
public:
    
    NeuralNetwork(): NeuralNetwork(DEFAULT_LAYERS, X_COLS, Y_COLS) {};
    NeuralNetwork(int nLayers, UINT iCount, UINT oCount);
    NeuralNetwork(int nLayers, std::vector<UINT> Units);
    virtual ~NeuralNetwork() {};
    
    VectorXf Calculate(VectorXf inputVector);
    
    void Backpropagate(VectorXf actualOutput, VectorXf desireOutput);
    
    void WriteWeight(std::string str = "/Users/liuyang/Desktop/Class/ML/DL/ANN/weight");
    
    void LoadWeight(std::string str = "/Users/liuyang/Desktop/Class/ML/DL/ANN/weight");
    
    void LoadWeight(std::vector<MatrixXf> weightMatrix, std::vector<VectorXf> biasVector);
    
private:
    const int nLayers;
    
    n_Neurons Neurons;
    
    UINT inputVectorDim;
    UINT outputVectorDim;
    
    VectorLayers m_Layers;
};

class NNLayer
{
public:
    NNLayer(int iLayerDim, int oLayerDim, NNLayer *pPrev = nullptr, bool loadWeight = false);
    virtual ~NNLayer() {};
    
    void Calculate();
    
    void Backpropagate(VectorXf &Err_dXn, VectorXf &Err_dXnm1,
                       double etaLearningRate);
    
    NNLayer* m_pPrevLayer;
    
    MatrixXf weightMatrix;      // Link the current layer and the previous
    VectorXf bias;
    
    VectorXf NeuronsValue;

};

double sigmoid(double input);

#endif
