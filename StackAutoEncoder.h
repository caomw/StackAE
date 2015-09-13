#ifndef __StackAE__StackAutoEncoder__
#define __StackAE__StackAutoEncoder__

#include "AutoEncoder.h"

#define DEFAULT_SPARSITY 0.05

typedef std::vector<AutoEncoder*> VectorAELayers;
typedef std::vector<unsigned int> HiddenUnits;

class StackAutoEncoder
{
public:
    StackAutoEncoder(const int layers,
                     const int rows, const int cols,
                     const int labels, const int testsamples);
    ~StackAutoEncoder();
    
    void LoadTrainData();
    void LoadTrainLabel();
    void LoadTestData();
    void LoadTestLabel();
    
    void FeedForward();
    
    void Test();
    
private:
    MatrixXf m_train_data;
    MatrixXf m_train_feature;
    MatrixXf m_train_label;
    
    MatrixXf m_test_data;
    MatrixXf m_test_label;
    MatrixXf m_test_feature;

    const int n_samples;
    const int n_layers;
    const int n_intput_dim;
    const int n_output_labels;
    const int n_test_samples;
    
    HiddenUnits m_hidden_units;
    VectorAELayers m_AELayers;
};

#endif /* defined(__StackAE__StackAutoEncoder__) */
