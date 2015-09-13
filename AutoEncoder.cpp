/*******************************************************************
 *  Copyright(c) 2015
 *  All rights reserved.
 *
 *  Name: Sparse/Basic Auto Encoder
 *  Description: A sparse model to find the interesting feature of data
 *
 *  Date: 2015-9-10
 *  Author: Yang
 *  Instruction: Use 10000 random 8 * 8 block of ten 512 * 512 image to
 *  train the model
 
 ******************************************************************/
#include "AutoEncoder.h"

AutoEncoder::AutoEncoder(std::string add,
                    const int rows, const int cols,UINT hiddenUnits,
                         double sparsity):
IO_dim(cols), hidden_units(hiddenUnits), samples(rows), st_sparsity(sparsity)
{
    
    /*
     *  Description:
     *  A constructed function to initialize the Sparse AutoEncoder
     *
     *  @param rows: number of samples
     *  @param cols: dimension of samples
     *  @param hiddenUnits: number of hidden units
     *  @param sparsity: standard sparsity we want to approximate
     *
     */

    LoadMatrix(*OriginalData, add, cols, rows);
    
    HiddenMatrix = MatrixXf::Zero(samples, hidden_units);
    Sparsity_average = VectorXf::Zero(hidden_units);
    OutputMatrix = MatrixXf::Zero(samples, IO_dim);
    
    /* Initialize the en/decode weight matrix */
    double w_init_interval = sqrt(6./(IO_dim + hidden_units + 1));
    Weight_encode = w_init_interval * MatrixXf::Random(hidden_units, IO_dim);
    Weight_decode = w_init_interval * MatrixXf::Random(IO_dim, hidden_units);
    
    Bias_encode = VectorXf::Ones(hidden_units);
    Bias_decode = VectorXf::Ones(IO_dim);
}

AutoEncoder::AutoEncoder(MatrixXf& matrix,
                         UINT hiddenUnits, double sparsity):
IO_dim(matrix.cols()), hidden_units(hiddenUnits), samples(matrix.rows()), st_sparsity(sparsity)
{
    OriginalData = &matrix;
    
    HiddenMatrix = MatrixXf::Zero(samples, hidden_units);
    Sparsity_average = VectorXf::Zero(hidden_units);
    OutputMatrix = MatrixXf::Zero(samples, IO_dim);
    
    /* Initialize the en/decode weight matrix */
    double w_init_interval = sqrt(6./(IO_dim + hidden_units + 1));
    Weight_encode = w_init_interval * MatrixXf::Random(hidden_units, IO_dim);
    Weight_decode = w_init_interval * MatrixXf::Random(IO_dim, hidden_units);
    
    Bias_encode = VectorXf::Ones(hidden_units);
    Bias_decode = VectorXf::Ones(IO_dim);
}

VectorXf AutoEncoder::Calculate(int index)
{
    /*
     *  Description:
     *  Calculate the i-th samples by feedforward and store hiddenvector
     *  in the row i of hidden matrix, output in row i of output matrix
     *
     *  @return outputVector: The output of FF
     */
    
    VectorXf HiddenVector = Weight_encode * OriginalData->row(index).transpose() + Bias_encode;
    for (int i = 0; i < HiddenVector.size(); i++)
    {
        HiddenVector(i) = sigmoid(HiddenVector(i));
    }
    HiddenMatrix.row(index) = HiddenVector.transpose();

    VectorXf output_vector = VectorXf(IO_dim);
    output_vector = Weight_decode * HiddenVector + Bias_decode;
    for (int i = 0; i < output_vector.size(); i++)
    {
        output_vector(i) = sigmoid(output_vector(i));
    }
    OutputMatrix.row(index) = output_vector.transpose();
    
    return output_vector;
}

void AutoEncoder::BackPropagate(int index, double etaLearningRate)
{
    /*
     *  Description:
     *  BP in the SAE and considering the cross-entropy cost
     *
     *  @param etaLearningRate: Step to convergence
     *
     */
    
    VectorXf delta_3th = OriginalData->row(index) - OutputMatrix.row(index) ;
    for (int i = 0; i < delta_3th.size(); i++)
    {
        delta_3th(i) = DSIGMOID(OutputMatrix(index, i)) * delta_3th(i);
    }

    VectorXf diff_2th = Weight_decode.transpose() * delta_3th;

    VectorXf delta_2th = VectorXf(diff_2th.size());
    /* If need sparse restriction */
    if (hidden_units >= IO_dim)
    {
        for (int i = 0; i < delta_2th.size(); i++)
        {
            double d_sp = -st_sparsity / Sparsity_average(i) +
            (1 - st_sparsity) / (1 - Sparsity_average(i));
            
            delta_2th(i) = DSIGMOID(HiddenMatrix(index, i)) * (diff_2th(i) + d_sp);
        }
    }
    else
    {
        for (int i = 0; i < delta_2th.size(); i++)
        {
            delta_2th(i) = DSIGMOID(HiddenMatrix(index, i)) * diff_2th(i);
        }
    }

    Weight_decode += etaLearningRate * delta_3th * HiddenMatrix.row(index);
    Weight_encode += etaLearningRate * delta_2th * OriginalData->row(index);
    Bias_decode += etaLearningRate * delta_3th;
    Bias_encode += etaLearningRate * delta_2th;
}

void AutoEncoder::Visualizing(int ImageCols, int ImageRows,
                              int FeatureCols, int FeatureRows)
{
    /*
     *  Description:
     *  Visualize the feature that hidden units read
     *
     *  @Param ImageCols: The number of features contained in the column
     *  @Param ImageRows: The number of features contained in the row
     *  @Param FeatureCols/Rows: The cols/rows of feature image.
     */
    
    cv::Mat image = cv::Mat(ImageRows * FeatureRows, ImageCols * FeatureCols, CV_8UC1);
    
    for (int i = 0; i < hidden_units; i++)
    {
        VectorXf out = Weight_encode.row(i) / sqrt((Weight_encode.row(i) * (Weight_encode.row(i).transpose())));
        for (int j = 0; j < FeatureRows; j++)
        {
            for (int k = 0; k < FeatureCols; k++)
            {
                image.at<uchar>(i / ImageCols * FeatureRows + j, i % ImageCols * FeatureCols + k) = uchar(int(out(j * FeatureCols + k) * 256));
            }
        }
    }
    cv::imshow("Feature", image);
    cv::waitKey();
}

void AutoEncoder::sparsity_averaging()
{
    Sparsity_average = VectorXf::Ones(samples).transpose() * HiddenMatrix;
    Sparsity_average = Sparsity_average / samples;
}
