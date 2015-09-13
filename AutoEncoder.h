#ifndef __AutoEncoder__AutoEncoder__
#define __AutoEncoder__AutoEncoder__

#include <random>
#include "ANN.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

class AutoEncoder
{
    using UINT = unsigned int;
    
public:
    
    AutoEncoder(std::string add, const int rows, const int cols,
                UINT hiddenUnits, double sparsity);
    AutoEncoder(MatrixXf& matrix,UINT hiddenUnits, double sparsity);
    
    virtual ~AutoEncoder() {};

    VectorXf Calculate(int index);
    
    void BackPropagate(int index, double etaLearningRate);
    void Visualizing(int ImageCols, int ImageRows,
                     int FeatureCols, int FeatureRows);
    
    void sparsity_averaging();
    
    MatrixXf& GetWeight_encode() {   return Weight_encode;   }
    MatrixXf& GetWeight_decode() {   return Weight_decode;   }
    VectorXf& GetBias_encode()   {   return Bias_encode;     }
    VectorXf& GetBias_decode()   {   return Bias_decode;     }
    MatrixXf& GetOriginalData()  {   return *OriginalData;   }
    MatrixXf& GetHiddenMatrix()  {   return HiddenMatrix;    }
    
private:
    
    MatrixXf *OriginalData;
    
    MatrixXf HiddenMatrix;
    MatrixXf OutputMatrix;
    VectorXf Sparsity_average;
    
    MatrixXf Weight_encode;
    MatrixXf Weight_decode;
    VectorXf Bias_encode;
    VectorXf Bias_decode;
    
    const long IO_dim;
    const long samples;
    
    double st_sparsity;
    
    UINT hidden_units;
    
};

#endif /* defined(__AutoEncoder__AutoEncoder__) */
