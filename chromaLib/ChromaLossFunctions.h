#pragma once

#include "ChromaBaseClasses.h"

namespace ChromaFlow {

class MSEParameterLoss {
public:
    // Mean squared error gradient for feature vectors
    // Returns dL/dy_pred = 2/N * (y_pred - y_true), as a FeatureTensor [1 x N]
    ChromaFlow::FeatureTensor calculate(const ChromaFlow::FeatureTensor& output,
                                        const ChromaFlow::FeatureTensor& target) const {
        const int N = output.data.cols() > 0 ? static_cast<int>(output.data.cols()) : 0;

        Eigen::VectorXf y_pred;
        if (output.data.rows() > 0) {
            y_pred = output.data.row(0).transpose(); // [N x 1]
        } else {
            y_pred = Eigen::VectorXf::Zero(static_cast<Eigen::Index>(N));
        }

        Eigen::VectorXf y_true;
        if (target.data.rows() > 0) {
            const int T = static_cast<int>(target.data.cols());
            // Pad or truncate target to match output length
            if (T == N) {
                y_true = target.data.row(0).transpose();
            } else if (T > N) {
                y_true = target.data.row(0).transpose().head(static_cast<Eigen::Index>(N));
            } else { // T < N
                y_true = Eigen::VectorXf::Zero(static_cast<Eigen::Index>(N));
                y_true.head(static_cast<Eigen::Index>(T)) = target.data.row(0).transpose();
            }
        } else {
            y_true = Eigen::VectorXf::Zero(static_cast<Eigen::Index>(N));
        }

        Eigen::VectorXf grad = Eigen::VectorXf::Zero(static_cast<Eigen::Index>(N));
        if (N > 0) {
            grad = (2.0f / static_cast<float>(N)) * (y_pred - y_true); // dL/dy
        }

        FeatureTensor outGrad;
        outGrad.data.resize(1, static_cast<Eigen::Index>(N));
        outGrad.data.row(0) = grad.transpose();
        outGrad.numSamples = 1;
        outGrad.features = N;
        return outGrad;
    }

};
} // namespace ChromaFlow